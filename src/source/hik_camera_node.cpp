#include "PixelType.h"
#include "MvCameraControl.h"
#include "cv_bridge/cv_bridge.h"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

struct CameraInfo {
    std::string serialNumber;
    std::string modelName;
    std::string ipAddress;
    unsigned int deviceType;
};

// 图像数据结构
struct ImageData {
    unsigned char *data;
    unsigned int width;
    unsigned int height;
    unsigned int pixelFormat;
    unsigned int dataSize;

    ImageData() : data(nullptr), width(0), height(0), pixelFormat(0), dataSize(0) {}
};

// 海康相机控制类
class CameraControl {
public:
    // 构造函数，初始化所有成员变量
    CameraControl() : handle_(nullptr), is_open_(false), is_grabbing_(false),
                     convert_buffer_(nullptr), buffer_size_(0), device_index_(0),
                     saved_exposure_(0.0f), saved_gain_(0.0f), saved_trigger_(false),
                     saved_frame_rate_(0.0f), saved_pixel_format_(0) {}

    ~CameraControl() {
        if (is_grabbing_) stop_grabbing();
        if (is_open_) close();
        if (convert_buffer_) delete[] convert_buffer_;
    }

    // 获取最后的错误信息
    std::string get_last_error() const {
        return last_error_;
    }

    // 重连相机
    bool reconnect(unsigned int index, int max_retries = 5, int retry_delay_ms = 1000) {
        device_index_ = index;

        for (int attempt = 1; attempt <= max_retries; ++attempt) {
            // 确保先关闭旧的连接
            if (is_grabbing_) stop_grabbing();
            if (is_open_) close();

            // 退避策略：重试间隔逐渐增加
            int current_delay = retry_delay_ms * attempt;
            std::this_thread::sleep_for(std::chrono::milliseconds(current_delay));

            // 尝试打开相机
            if (!open(index)) {
                RCLCPP_DEBUG(rclcpp::get_logger("camera_driver"),
                           "重连尝试 %d/%d 失败: %s", attempt, max_retries, get_last_error().c_str());
                continue;
            }

            // 恢复设置
            bool settings_ok = true;

            if (saved_pixel_format_ > 0 && !set_pixel_format(saved_pixel_format_)) {
                settings_ok = false;
            }

            if (saved_exposure_ > 0.0f && !set_exposure(saved_exposure_)) {
                settings_ok = false;
            }

            if (saved_gain_ > 0.0f && !set_gain(saved_gain_)) {
                settings_ok = false;
            }

            if (saved_frame_rate_ > 0.0f && !set_frame_rate(saved_frame_rate_)) {
                settings_ok = false;
            }

            if (!set_trigger_mode(saved_trigger_)) {
                settings_ok = false;
            }

            // 尝试开始采集
            if (!start_grabbing()) {
                RCLCPP_DEBUG(rclcpp::get_logger("camera_driver"),
                           "重连采集失败: %s", get_last_error().c_str());
                close();
                continue;
            }

            RCLCPP_DEBUG(rclcpp::get_logger("camera_driver"),
                       "重连尝试 %d/%d 成功", attempt, max_retries);
            return true;
        }

        RCLCPP_WARN(rclcpp::get_logger("camera_driver"),
                    "重连失败: 经过 %d 次尝试后仍无法连接", max_retries);
        return false;
    }

    // 软件触发
    bool trigger_software() {
        if (!is_open_) return false;
        int ret = MV_CC_SetCommandValue(handle_, "TriggerSoftware");
        return ret == MV_OK;
    }

    // 获取图像宽度
    unsigned int get_width() {
        if (!is_open_) return 0;
        MVCC_INTVALUE value;
        return MV_CC_GetIntValue(handle_, "Width", &value) == MV_OK ? value.nCurValue : 0;
    }

    // 获取图像高度
    unsigned int get_height() {
        if (!is_open_) return 0;
        MVCC_INTVALUE value;
        return MV_CC_GetIntValue(handle_, "Height", &value) == MV_OK ? value.nCurValue : 0;
    }

    static std::vector<CameraInfo> enumerate_devices() {
        std::vector<CameraInfo> devices;
        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) return devices;

        for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
            MV_CC_DEVICE_INFO *info = device_list.pDeviceInfo[i];
            if (!info) continue;

            CameraInfo cam_info;
            cam_info.deviceType = info->nTLayerType;

            if (info->nTLayerType == MV_GIGE_DEVICE) {
                auto *gige_info = &info->SpecialInfo.stGigEInfo;
                cam_info.serialNumber = std::string((char *)gige_info->chSerialNumber);
                cam_info.modelName = std::string((char *)gige_info->chModelName);

                unsigned int ip = gige_info->nCurrentIp;
                std::ostringstream oss;
                oss << ((ip & 0xFF000000) >> 24) << "." << ((ip & 0x00FF0000) >> 16)
                    << "." << ((ip & 0x0000FF00) >> 8) << "." << (ip & 0x000000FF);
                cam_info.ipAddress = oss.str();
            } else if (info->nTLayerType == MV_USB_DEVICE) {
                auto *usb_info = &info->SpecialInfo.stUsb3VInfo;
                cam_info.serialNumber = std::string((char *)usb_info->chSerialNumber);
                cam_info.modelName = std::string((char *)usb_info->chModelName);
                cam_info.ipAddress = "USB";
            }
            devices.push_back(cam_info);
        }
        return devices;
    }

    // 打开指定索引的相机设备
    bool open(unsigned int index = 0) {
        if (is_open_) {
            last_error_ = "Camera is already open";
            return false;
        }

        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        // 重新扫描设备列表
        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) {
            set_error("Enumerate devices failed", ret);
            return false;
        }
        if (index >= device_list.nDeviceNum) {
            last_error_ = "Invalid device index";
            return false;
        }

        // 创建设备句柄
        ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[index]);
        if (ret != MV_OK) {
            set_error("Create handle failed", ret);
            return false;
        }

        // 打开设备
        ret = MV_CC_OpenDevice(handle_);
        if (ret != MV_OK) {
            set_error("Open device failed", ret);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
            return false;
        }

        // 设置触发模式为关闭（连续采集模式）
        ret = MV_CC_SetEnumValue(handle_, "TriggerMode", 0);
        if (ret != MV_OK) {
            set_error("Set trigger mode failed", ret);
        }

        is_open_ = true;
        device_index_ = index;
        return true;
    }

    bool close() {
        if (!is_open_) return true;
        if (is_grabbing_) stop_grabbing();

        int ret = MV_CC_CloseDevice(handle_);
        if (ret != MV_OK) {
            set_error("Close device failed", ret);
        }

        ret = MV_CC_DestroyHandle(handle_);
        if (ret != MV_OK) {
            set_error("Destroy handle failed", ret);
        }

        handle_ = nullptr;
        is_open_ = false;
        return true;
    }

    bool start_grabbing() {
        if (!is_open_) {
            last_error_ = "Camera is not open";
            return false;
        }
        if (is_grabbing_) return true;

        int ret = MV_CC_StartGrabbing(handle_);
        if (ret != MV_OK) {
            set_error("Start grabbing failed", ret);
            return false;
        }
        is_grabbing_ = true;
        return true;
    }

    bool stop_grabbing() {
        if (!is_grabbing_) return true;
        int ret = MV_CC_StopGrabbing(handle_);
        if (ret != MV_OK) {
            set_error("Stop grabbing failed", ret);
            return false;
        }
        is_grabbing_ = false;
        return true;
    }

    bool grab_image(ImageData &data, unsigned int timeout = 1000, unsigned int desired_format = 0) {
        if (!is_grabbing_) {
            last_error_ = "相机未开始采集";
            return false;
        }

        MV_FRAME_OUT frame;
        memset(&frame, 0, sizeof(MV_FRAME_OUT));

        int ret = MV_CC_GetImageBuffer(handle_, &frame, timeout);
        if (ret != MV_OK) {
            set_error("获取图像缓冲区失败", ret);
            return false;
        }

        // 检查图像数据有效性
        if (!frame.pBufAddr || frame.stFrameInfo.nWidth == 0 || frame.stFrameInfo.nHeight == 0) {
            MV_CC_FreeImageBuffer(handle_, &frame);
            last_error_ = "获取到无效图像数据";
            return false;
        }

        unsigned int width = frame.stFrameInfo.nWidth;
        unsigned int height = frame.stFrameInfo.nHeight;
        unsigned int src_format = frame.stFrameInfo.enPixelType;
        unsigned int target_format = desired_format ? desired_format : src_format;
        bool need_convert = (target_format != src_format);

        // 验证目标格式是否受支持
        if (need_convert && !is_pixel_format_supported(target_format)) {
            RCLCPP_WARN(rclcpp::get_logger("camera_driver"),
                       "目标像素格式 0x%08X 不受支持，使用原始格式", target_format);
            target_format = src_format;
            need_convert = false;
        }

        size_t required_size = need_convert ? estimate_buffer_size(target_format, width, height)
                                           : frame.stFrameInfo.nFrameLen;
        if (required_size == 0) {
            MV_CC_FreeImageBuffer(handle_, &frame);
            last_error_ = "无法计算所需缓冲区大小";
            return false;
        }

        // 智能缓冲区管理：避免频繁重新分配
        if (buffer_size_ < required_size) {
            if (convert_buffer_) {
                delete[] convert_buffer_;
                convert_buffer_ = nullptr;
            }
            try {
                convert_buffer_ = new unsigned char[required_size];
                buffer_size_ = required_size;
                RCLCPP_DEBUG(rclcpp::get_logger("camera_driver"),
                           "分配新的转换缓冲区: %u 字节", buffer_size_);
            } catch (const std::bad_alloc&) {
                MV_CC_FreeImageBuffer(handle_, &frame);
                last_error_ = "内存分配失败: 缓冲区大小 " + std::to_string(required_size);
                return false;
            }
        }

        if (need_convert) {
            MV_CC_PIXEL_CONVERT_PARAM convert_param;
            memset(&convert_param, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));
            convert_param.nWidth = width;
            convert_param.nHeight = height;
            convert_param.pSrcData = frame.pBufAddr;
            convert_param.nSrcDataLen = frame.stFrameInfo.nFrameLen;
            convert_param.enSrcPixelType = static_cast<MvGvspPixelType>(src_format);
            convert_param.enDstPixelType = static_cast<MvGvspPixelType>(target_format);
            convert_param.pDstBuffer = convert_buffer_;
            convert_param.nDstBufferSize = buffer_size_;

            ret = MV_CC_ConvertPixelType(handle_, &convert_param);
            if (ret == MV_OK) {
                data.pixelFormat = target_format;
                data.data = convert_buffer_;
                data.dataSize = convert_param.nDstLen;
            } else {
                set_error("像素格式转换失败", ret);
                MV_CC_FreeImageBuffer(handle_, &frame);
                return false;
            }
        } else {
            // 即使不需要转换，也复制到内部缓冲区以确保数据安全
            if (buffer_size_ < frame.stFrameInfo.nFrameLen) {
                delete[] convert_buffer_;
                try {
                    convert_buffer_ = new unsigned char[frame.stFrameInfo.nFrameLen];
                    buffer_size_ = frame.stFrameInfo.nFrameLen;
                } catch (const std::bad_alloc&) {
                    MV_CC_FreeImageBuffer(handle_, &frame);
                    last_error_ = "内存分配失败";
                    return false;
                }
            }
            memcpy(convert_buffer_, frame.pBufAddr, frame.stFrameInfo.nFrameLen);
            data.pixelFormat = src_format;
            data.data = convert_buffer_;
            data.dataSize = frame.stFrameInfo.nFrameLen;
        }

        data.width = width;
        data.height = height;
        MV_CC_FreeImageBuffer(handle_, &frame);

        RCLCPP_DEBUG(rclcpp::get_logger("camera_driver"),
                   "成功获取图像: %ux%u, 格式: 0x%08X, 大小: %u 字节",
                   data.width, data.height, data.pixelFormat, data.dataSize);

        return true;
    }

    bool set_exposure(float exposure) {
        if (!is_open_) {
            last_error_ = "Camera is not open";
            return false;
        }
        int ret = MV_CC_SetFloatValue(handle_, "ExposureTime", exposure);
        if (ret != MV_OK) {
            set_error("Set exposure time failed", ret);
            return false;
        }
        saved_exposure_ = exposure;
        return true;
    }

    float get_exposure() {
        if (!is_open_) return 0.0f;
        MVCC_FLOATVALUE value;
        return MV_CC_GetFloatValue(handle_, "ExposureTime", &value) == MV_OK ? value.fCurValue : 0.0f;
    }

    bool set_gain(float gain) {
        if (!is_open_) return false;
        int ret = MV_CC_SetFloatValue(handle_, "Gain", gain);
        if (ret != MV_OK) return false;
        saved_gain_ = gain;
        return true;
    }

    float get_gain() {
        if (!is_open_) return 0.0f;
        MVCC_FLOATVALUE value;
        return MV_CC_GetFloatValue(handle_, "Gain", &value) == MV_OK ? value.fCurValue : 0.0f;
    }

    bool set_trigger_mode(bool enable) {
        if (!is_open_) return false;
        int ret = MV_CC_SetEnumValue(handle_, "TriggerMode", enable ? 1 : 0);
        if (ret != MV_OK) return false;
        saved_trigger_ = enable;
        return true;
    }

    bool set_frame_rate(float fps) {
        if (!is_open_) return false;
        if (fps <= 0.0f) return false;

        MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
        MV_CC_SetEnumValue(handle_, "AcquisitionFrameRateAuto", 0);

        int ret = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", fps);
        if (ret != MV_OK) {
            MVCC_FLOATVALUE limits;
            if (MV_CC_GetFloatValue(handle_, "AcquisitionFrameRate", &limits) == MV_OK) {
                float clamped = std::clamp(fps, limits.fMin, limits.fMax);
                ret = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", clamped);
            }
        }

        if (ret == MV_OK) {
            saved_frame_rate_ = fps;
            return true;
        }
        return false;
    }

    float get_frame_rate() {
        if (!is_open_) return 0.0f;
        MVCC_FLOATVALUE value;
        // 返回设置的帧率而不是实际帧率，这样能反映用户的设置意图
        if (MV_CC_GetFloatValue(handle_, "AcquisitionFrameRate", &value) == MV_OK) return value.fCurValue;
        return saved_frame_rate_;
    }

    float get_actual_frame_rate() {
        if (!is_open_) return 0.0f;
        MVCC_FLOATVALUE value;
        if (MV_CC_GetFloatValue(handle_, "ResultingFrameRate", &value) == MV_OK) return value.fCurValue;
        return 0.0f;
    }

    bool set_pixel_format(unsigned int format) {
        if (!is_open_) return false;
        if (saved_pixel_format_ == format && format != 0) return true;

        // 验证像素格式是否受支持
        if (!is_pixel_format_supported(format)) {
            last_error_ = "不支持的像素格式: 0x" + std::to_string(format);
            return false;
        }

        bool was_grabbing = is_grabbing_;
        if (was_grabbing) stop_grabbing();

        int ret = MV_CC_SetEnumValue(handle_, "PixelFormat", format);
        if (ret != MV_OK) {
            set_error("设置像素格式失败", ret);
            if (was_grabbing) start_grabbing();
            return false;
        }

        saved_pixel_format_ = format;
        if (was_grabbing) start_grabbing();

        RCLCPP_INFO(rclcpp::get_logger("camera_driver"), "像素格式已设置为: 0x%08X", format);
        return true;
    }

    // 检查像素格式是否受支持
    bool is_pixel_format_supported(unsigned int format) const {
        // 支持的像素格式列表
        static const std::vector<unsigned int> supported_formats = {
            PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono12,
            PixelType_Gvsp_Mono14, PixelType_Gvsp_Mono16,
            PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_RGB10_Packed, PixelType_Gvsp_BGR10_Packed,
            PixelType_Gvsp_RGB12_Packed, PixelType_Gvsp_BGR12_Packed,
            PixelType_Gvsp_RGB16_Packed, PixelType_Gvsp_BGR16_Packed,
            PixelType_Gvsp_RGBA8_Packed, PixelType_Gvsp_BGRA8_Packed,
            PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed,
            PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerRG10,
            PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerBG10,
            PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerRG12,
            PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerBG12,
            PixelType_Gvsp_BayerGR16, PixelType_Gvsp_BayerRG16,
            PixelType_Gvsp_BayerGB16, PixelType_Gvsp_BayerBG16
        };

        return std::find(supported_formats.begin(), supported_formats.end(), format) != supported_formats.end();
    }

    // 获取支持的像素格式列表
    std::vector<unsigned int> get_supported_pixel_formats() const {
        return {
            PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono12,
            PixelType_Gvsp_Mono14, PixelType_Gvsp_Mono16,
            PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_RGB10_Packed, PixelType_Gvsp_BGR10_Packed,
            PixelType_Gvsp_RGB12_Packed, PixelType_Gvsp_BGR12_Packed,
            PixelType_Gvsp_RGB16_Packed, PixelType_Gvsp_BGR16_Packed,
            PixelType_Gvsp_RGBA8_Packed, PixelType_Gvsp_BGRA8_Packed,
            PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed,
            PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerRG10,
            PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerBG10,
            PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerRG12,
            PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerBG12,
            PixelType_Gvsp_BayerGR16, PixelType_Gvsp_BayerRG16,
            PixelType_Gvsp_BayerGB16, PixelType_Gvsp_BayerBG16
        };
    }

    // 获取像素格式的名称描述
    std::string get_pixel_format_name(unsigned int format) const {
        switch (format) {
        case PixelType_Gvsp_Mono8: return "Mono8";
        case PixelType_Gvsp_Mono10: return "Mono10";
        case PixelType_Gvsp_Mono12: return "Mono12";
        case PixelType_Gvsp_Mono14: return "Mono14";
        case PixelType_Gvsp_Mono16: return "Mono16";
        case PixelType_Gvsp_RGB8_Packed: return "RGB8";
        case PixelType_Gvsp_BGR8_Packed: return "BGR8";
        case PixelType_Gvsp_RGB10_Packed: return "RGB10";
        case PixelType_Gvsp_BGR10_Packed: return "BGR10";
        case PixelType_Gvsp_RGB12_Packed: return "RGB12";
        case PixelType_Gvsp_BGR12_Packed: return "BGR12";

        case PixelType_Gvsp_RGB16_Packed: return "RGB16";
        case PixelType_Gvsp_BGR16_Packed: return "BGR16";
        case PixelType_Gvsp_RGBA8_Packed: return "RGBA8";
        case PixelType_Gvsp_BGRA8_Packed: return "BGRA8";
        case PixelType_Gvsp_YUV422_Packed: return "YUV422";
        case PixelType_Gvsp_YUV422_YUYV_Packed: return "YUV422_YUYV";
        case PixelType_Gvsp_BayerGR8: return "BayerGR8";
        case PixelType_Gvsp_BayerRG8: return "BayerRG8";
        case PixelType_Gvsp_BayerGB8: return "BayerGB8";
        case PixelType_Gvsp_BayerBG8: return "BayerBG8";
        case PixelType_Gvsp_BayerGR10: return "BayerGR10";
        case PixelType_Gvsp_BayerRG10: return "BayerRG10";
        case PixelType_Gvsp_BayerGB10: return "BayerGB10";
        case PixelType_Gvsp_BayerBG10: return "BayerBG10";
        case PixelType_Gvsp_BayerGR12: return "BayerGR12";
        case PixelType_Gvsp_BayerRG12: return "BayerRG12";
        case PixelType_Gvsp_BayerGB12: return "BayerGB12";
        case PixelType_Gvsp_BayerBG12: return "BayerBG12";
        case PixelType_Gvsp_BayerGR16: return "BayerGR16";
        case PixelType_Gvsp_BayerRG16: return "BayerRG16";
        case PixelType_Gvsp_BayerGB16: return "BayerGB16";
        case PixelType_Gvsp_BayerBG16: return "BayerBG16";
        default: return "Unknown";
        }
    }

    bool is_open() const { return is_open_; }
    bool is_grabbing() const { return is_grabbing_; }

    // 新增功能：通过序列号打开相机
    bool open_by_serial_number(const std::string &serial_number) {
        if (is_open_) return false;

        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) {
            set_error("Enumerate devices failed", ret);
            return false;
        }

        // 查找指定序列号的设备
        int device_index = -1;
        for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
            MV_CC_DEVICE_INFO *info = device_list.pDeviceInfo[i];
            std::string sn;

            if (info->nTLayerType == MV_GIGE_DEVICE) {
                sn = std::string((char *)info->SpecialInfo.stGigEInfo.chSerialNumber);
            } else if (info->nTLayerType == MV_USB_DEVICE) {
                sn = std::string((char *)info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            }

            if (sn == serial_number) {
                device_index = i;
                break;
            }
        }

        if (device_index < 0) {
            last_error_ = "Device with serial number '" + serial_number + "' not found";
            return false;
        }
        return open(device_index);
    }



private:
    void *handle_;
    bool is_open_;
    bool is_grabbing_;
    unsigned char *convert_buffer_;
    unsigned int buffer_size_;
    unsigned int device_index_;
    float saved_exposure_;
    float saved_gain_;
    bool saved_trigger_;
    float saved_frame_rate_;
    unsigned int saved_pixel_format_;
    std::string last_error_;

    size_t estimate_buffer_size(unsigned int format, unsigned int w, unsigned int h) {
        switch (format) {
        case PixelType_Gvsp_Mono8: return w * h;
        case PixelType_Gvsp_Mono10:
        case PixelType_Gvsp_Mono12:
        case PixelType_Gvsp_Mono14:
        case PixelType_Gvsp_Mono16: return w * h * 2;
        case PixelType_Gvsp_RGB8_Packed:
        case PixelType_Gvsp_BGR8_Packed: return w * h * 3;
        case PixelType_Gvsp_RGB10_Packed:
        case PixelType_Gvsp_BGR10_Packed:
        case PixelType_Gvsp_RGB12_Packed:
        case PixelType_Gvsp_BGR12_Packed:
        case PixelType_Gvsp_RGB16_Packed:
        case PixelType_Gvsp_BGR16_Packed: return w * h * 6;
        case PixelType_Gvsp_RGBA8_Packed:
        case PixelType_Gvsp_BGRA8_Packed: return w * h * 4;
        case PixelType_Gvsp_YUV422_Packed:
        case PixelType_Gvsp_YUV422_YUYV_Packed: return w * h * 2;
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_BayerRG8:
        case PixelType_Gvsp_BayerGB8:
        case PixelType_Gvsp_BayerBG8: return w * h;
        case PixelType_Gvsp_BayerGR10:
        case PixelType_Gvsp_BayerRG10:
        case PixelType_Gvsp_BayerGB10:
        case PixelType_Gvsp_BayerBG10:
        case PixelType_Gvsp_BayerGR12:
        case PixelType_Gvsp_BayerRG12:
        case PixelType_Gvsp_BayerGB12:
        case PixelType_Gvsp_BayerBG12:
        case PixelType_Gvsp_BayerGR16:
        case PixelType_Gvsp_BayerRG16:
        case PixelType_Gvsp_BayerGB16:
        case PixelType_Gvsp_BayerBG16: return w * h * 2;
        default: return w * h * 3;
        }
    }

    // 设置错误信息
    void set_error(const std::string &error, int error_code) {
        std::ostringstream oss;
        oss << error << " (Error code: 0x" << std::hex << error_code << ")";
        last_error_ = oss.str();
    }
};

// ROS2节点类
class CameraNode : public rclcpp::Node {
public:
    explicit CameraNode() : Node("camera_driver") {
        declare_parameter("exposure", 4000.0);
        declare_parameter("image_gain", 16.9807);
        declare_parameter("use_trigger", false);
        declare_parameter("fps", 165.0);
        declare_parameter("pixel_format_code", static_cast<int>(PixelType_Gvsp_BayerRG8));
        declare_parameter("camera_frame", "camera_optical_frame");
        declare_parameter("serial_number", "");

        exposure_ = get_parameter("exposure").as_double();
        gain_ = get_parameter("image_gain").as_double();
        trigger_ = get_parameter("use_trigger").as_bool();
        frame_rate_ = get_parameter("fps").as_double();
        pixel_format_ = get_parameter("pixel_format_code").as_int();
        frame_id_ = get_parameter("camera_frame").as_string();
        serial_number_ = get_parameter("serial_number").as_string();

        param_callback_ = add_on_set_parameters_callback([this](const auto &params) {
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            std::ostringstream reason;

            // 遍历所有参数变更
            for (const auto &p : params) {
                if (p.get_name() == "exposure") {
                    double v = p.as_double();
                    double previous = exposure_;
                    if (!camera_.set_exposure(v)) {
                        result.successful = false;
                        exposure_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "曝光时间设置失败: " << camera_.get_last_error();
                    } else {
                        exposure_ = v;
                    }
                } else if (p.get_name() == "image_gain") {
                    double v = p.as_double();
                    double previous = gain_;
                    if (!camera_.set_gain(v)) {
                        result.successful = false;
                        gain_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "增益设置失败: " << camera_.get_last_error();
                    } else {
                        gain_ = v;
                    }
                } else if (p.get_name() == "use_trigger") {
                    bool v = p.as_bool();
                    bool previous = trigger_;
                    if (!camera_.set_trigger_mode(v)) {
                        result.successful = false;
                        trigger_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "触发模式设置失败: " << camera_.get_last_error();
                    } else {
                        trigger_ = v;
                    }
                } else if (p.get_name() == "fps") {
                    double v = p.as_double();
                    if (v <= 0.0) {
                        result.successful = false;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "帧率必须大于0";
                        continue;
                    }
                    double previous = frame_rate_;
                    if (!camera_.set_frame_rate(v)) {
                        result.successful = false;
                        frame_rate_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "帧率设置失败: " << camera_.get_last_error();
                    } else {
                        frame_rate_ = v;
                    }
                } else if (p.get_name() == "pixel_format_code") {
                    int v = p.as_int();
                    unsigned int previous = pixel_format_;

                    // 验证像素格式是否受支持
                    if (!camera_.is_pixel_format_supported(v)) {
                        result.successful = false;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "像素格式 0x" << std::hex << v << " 不受支持";
                        continue;
                    }

                    if (!camera_.set_pixel_format(v)) {
                        result.successful = false;
                        pixel_format_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "像素格式设置失败: " << camera_.get_last_error();
                    } else {
                        pixel_format_ = v;
                        RCLCPP_INFO(get_logger(), "像素格式已更新为: 0x%08X", v);
                    }
                } else if (p.get_name() == "camera_frame") {
                    frame_id_ = p.as_string();
                } else if (p.get_name() == "serial_number") {
                    std::string v = p.as_string();
                    if (v != serial_number_) {
                        if (camera_.is_open()) {
                            RCLCPP_WARN(get_logger(), "序列号变更需要重启节点");
                        }
                        serial_number_ = v;
                    }
                }
            }
            result.reason = reason.str();
            return result;
        });

        publisher_ = create_publisher<sensor_msgs::msg::Image>("/image_raw", 10);
        timer_ = create_wall_timer(2ms, std::bind(&CameraNode::publish_image, this));
        last_fps_time_ = now();
    }

    ~CameraNode() {
        camera_.close();
    }

private:
    void publish_image() {
        // 如果相机未连接，尝试重连
        if (!camera_.is_open()) {
            bool is_initial_connect = !last_attempt_valid_;

            if (!settings_applied_ || !last_attempt_valid_ || (now() - last_attempt_).seconds() >= reconnect_interval_sec_) {
                last_attempt_ = now();
                last_attempt_valid_ = true;
                bool success = false;

                if (is_initial_connect) {
                    RCLCPP_INFO(get_logger(), "正在连接相机...");
                } else {
                    RCLCPP_WARN(get_logger(), "相机连接断开，尝试重连...");
                }

                if (!serial_number_.empty()) {
                    // 使用序列号打开相机
                    success = camera_.open_by_serial_number(serial_number_);
                    if (success) {
                        bool settings_ok = apply_settings();
                        if (!camera_.start_grabbing()) {
                            RCLCPP_WARN(get_logger(), "开始采集失败: %s", camera_.get_last_error().c_str());
                            camera_.close();
                            settings_applied_ = false;
                            last_attempt_valid_ = false;
                            success = false;
                        } else if (!settings_ok) {
                            RCLCPP_WARN(get_logger(), "相机连接成功，但部分参数应用失败");
                        }
                    }
                } else {
                    // 使用设备索引重连
                    success = camera_.reconnect(0, 3, 500);
                    if (success) {
                        apply_settings();
                    }
                }

                if (success) {
                    if (is_initial_connect) {
                        RCLCPP_INFO(get_logger(), "相机连接成功");
                    } else {
                        RCLCPP_INFO(get_logger(), "相机重连成功");
                    }
                } else {
                    RCLCPP_WARN(get_logger(), "连接失败: %s", camera_.get_last_error().c_str());
                }
            }
            return;
        }

        // 如果设置未应用，应用相机设置
        if (!settings_applied_) {
            bool settings_ok = apply_settings();
            if (!settings_ok) {
                RCLCPP_WARN(get_logger(), "部分相机参数应用失败，将继续尝试");
            }
        }

        // 如果相机未开始采集，开始采集
        if (!camera_.is_grabbing()) {
            if (!camera_.start_grabbing()) {
                RCLCPP_WARN(get_logger(), "开始采集失败: %s", camera_.get_last_error().c_str());
                camera_.close();
                settings_applied_ = false;
                return;
            }
        }

        // 采集图像数据
        ImageData img;
        if (!camera_.grab_image(img, 1000, pixel_format_)) {
            consecutive_failures_++;

            // 使用 throttle 日志避免日志刷屏
            auto now_time = now();
            auto throttle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(2)).count();

            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), throttle_ns,
                               "获取图像失败 (连续失败: %u次): %s",
                               consecutive_failures_, camera_.get_last_error().c_str());

            // 连续失败次数过多，可能是连接断开，强制关闭并重连
            if (consecutive_failures_ >= max_grab_failures_) {
                RCLCPP_WARN(get_logger(), "连续获取图像失败 %u 次，关闭相机以触发重连",
                           consecutive_failures_);
                camera_.close();
                settings_applied_ = false;
                last_attempt_valid_ = false;
                consecutive_failures_ = 0;
            }

            if (!camera_.is_open()) {
                settings_applied_ = false;
                last_attempt_valid_ = false;
                consecutive_failures_ = 0;
            }
            return;
        }

        // 成功获取图像，重置连续失败计数器
        consecutive_failures_ = 0;

        // 检查图像数据有效性
        if (!img.data || !img.width || !img.height) return;

        std::string encoding = to_encoding(img.pixelFormat);
        cv::Mat frame;

        // 根据像素格式创建对应的Mat对象
        if (encoding == sensor_msgs::image_encodings::MONO16 ||
            encoding == sensor_msgs::image_encodings::RGB16) {
            frame = cv::Mat(img.height, img.width, CV_16UC1, img.data);
        } else {
            frame = cv::Mat(img.height, img.width, CV_8UC1, img.data);
        }

        // Bayer格式去马赛克处理
        if (sensor_msgs::image_encodings::isBayer(encoding)) {
            cv::Mat bgr;

            // 根据不同的Bayer模式选择对应的去马赛克算法
            // 由于颜色显示异常（黄色变蓝色、蓝色变橘色），可能是R和B通道互换
            if (encoding == sensor_msgs::image_encodings::BAYER_RGGB8 ||
                encoding == sensor_msgs::image_encodings::BAYER_RGGB16) {
                cv::demosaicing(frame, bgr, cv::COLOR_BayerRG2BGR);
            } else if (encoding == sensor_msgs::image_encodings::BAYER_GRBG8 ||
                       encoding == sensor_msgs::image_encodings::BAYER_GRBG16) {
                cv::demosaicing(frame, bgr, cv::COLOR_BayerGR2BGR);
            } else if (encoding == sensor_msgs::image_encodings::BAYER_GBRG8 ||
                       encoding == sensor_msgs::image_encodings::BAYER_GBRG16) {
                cv::demosaicing(frame, bgr, cv::COLOR_BayerGB2BGR);
            } else if (encoding == sensor_msgs::image_encodings::BAYER_BGGR8 ||
                       encoding == sensor_msgs::image_encodings::BAYER_BGGR16) {
                cv::demosaicing(frame, bgr, cv::COLOR_BayerBG2BGR);
            } else {
                RCLCPP_WARN(get_logger(), "未知Bayer格式: %s，使用默认RGGB到BGR模式", encoding.c_str());
                cv::demosaicing(frame, bgr, cv::COLOR_BayerRG2BGR);
            }

            // 手动交换R和B通道以修复颜色问题
            std::vector<cv::Mat> channels;
            cv::split(bgr, channels);
            std::swap(channels[0], channels[2]); // 交换B和R通道
            cv::merge(channels, bgr);

            frame = bgr;
            // 更新编码格式为BGR格式
            encoding = (encoding.find("16") != std::string::npos) ?
                       sensor_msgs::image_encodings::BGR16 :
                       sensor_msgs::image_encodings::BGR8;
            RCLCPP_DEBUG(get_logger(), "Bayer图像已转换并交换R/B通道: %s", encoding.c_str());
        }

        // YUV422格式转换
        else if (encoding == sensor_msgs::image_encodings::YUV422) {
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_YUV2RGB_YUYV);
            frame = rgb;
            encoding = sensor_msgs::image_encodings::RGB8;
            RCLCPP_DEBUG(get_logger(), "YUV422图像已转换为RGB格式");
        }

        auto header = std_msgs::msg::Header();
        header.stamp = now();
        header.frame_id = frame_id_;

        auto msg = cv_bridge::CvImage(header, encoding, frame).toImageMsg();
        publisher_->publish(*msg);

        // 显示帧率（每3秒显示一次）
        auto current_time = now();
        if ((current_time - last_fps_time_).seconds() >= 3.0) {
            double actual_fps = camera_.get_actual_frame_rate();
            RCLCPP_INFO(get_logger(), "FPS设置: %.2f, 实际: %.2f, Size: %dx%d",
                       frame_rate_, actual_fps, img.width, img.height);
            last_fps_time_ = current_time;
        }
    }

    // 应用所有相机参数设置
    bool apply_settings() {
        if (!camera_.is_open()) return false;

        bool ok = true;

        // 先设置像素格式，因为其他设置可能依赖于像素格式
        if (camera_.is_pixel_format_supported(pixel_format_)) {
            if (!camera_.set_pixel_format(pixel_format_)) {
                ok = false;
                RCLCPP_WARN(get_logger(), "像素格式设置失败: %s", camera_.get_last_error().c_str());
            }
        } else {
            RCLCPP_WARN(get_logger(), "像素格式 0x%08X 不受支持，使用默认格式", pixel_format_);
            // 尝试设置一个默认的受支持格式
            unsigned int default_format = PixelType_Gvsp_BayerRG8;
            if (camera_.set_pixel_format(default_format)) {
                pixel_format_ = default_format;
            } else {
                ok = false;
                RCLCPP_WARN(get_logger(), "默认像素格式设置失败");
            }
        }

        if (!camera_.set_exposure(exposure_)) {
            ok = false;
            RCLCPP_DEBUG(get_logger(), "曝光时间设置失败: %s", camera_.get_last_error().c_str());
        }

        if (!camera_.set_gain(gain_)) {
            ok = false;
            RCLCPP_DEBUG(get_logger(), "增益设置失败: %s", camera_.get_last_error().c_str());
        }

        if (!camera_.set_trigger_mode(trigger_)) {
            ok = false;
            RCLCPP_DEBUG(get_logger(), "触发模式设置失败: %s", camera_.get_last_error().c_str());
        }

        if (frame_rate_ > 0.0) {
            bool success = camera_.set_frame_rate(frame_rate_);
            if (!success) {
                ok = false;
                RCLCPP_DEBUG(get_logger(), "帧率设置失败: %s", camera_.get_last_error().c_str());
            } else {
                double applied = static_cast<double>(camera_.get_frame_rate());
                if (applied > 0.0) {
                    frame_rate_ = applied;
                }
            }
        }

        settings_applied_ = ok;
        return ok;
    }

    std::string to_encoding(unsigned int format) const {
        switch (format) {
        case PixelType_Gvsp_Mono8: return sensor_msgs::image_encodings::MONO8;
        case PixelType_Gvsp_Mono10:
        case PixelType_Gvsp_Mono12:
        case PixelType_Gvsp_Mono14:
        case PixelType_Gvsp_Mono16: return sensor_msgs::image_encodings::MONO16;
        case PixelType_Gvsp_RGB8_Packed: return sensor_msgs::image_encodings::RGB8;
        case PixelType_Gvsp_BGR8_Packed: return sensor_msgs::image_encodings::BGR8;
        case PixelType_Gvsp_RGB10_Packed:
        case PixelType_Gvsp_BGR10_Packed:
        case PixelType_Gvsp_RGB12_Packed:
        case PixelType_Gvsp_BGR12_Packed:
        case PixelType_Gvsp_RGB16_Packed:
        case PixelType_Gvsp_BGR16_Packed: return sensor_msgs::image_encodings::RGB16;
        case PixelType_Gvsp_RGBA8_Packed: return sensor_msgs::image_encodings::RGBA8;
        case PixelType_Gvsp_BGRA8_Packed: return sensor_msgs::image_encodings::BGRA8;
        case PixelType_Gvsp_YUV422_Packed:
        case PixelType_Gvsp_YUV422_YUYV_Packed: return sensor_msgs::image_encodings::YUV422;
        case PixelType_Gvsp_BayerGR8: return sensor_msgs::image_encodings::BAYER_GRBG8;
        case PixelType_Gvsp_BayerRG8: return sensor_msgs::image_encodings::BAYER_RGGB8;
        case PixelType_Gvsp_BayerGB8: return sensor_msgs::image_encodings::BAYER_GBRG8;
        case PixelType_Gvsp_BayerBG8: return sensor_msgs::image_encodings::BAYER_BGGR8;
        case PixelType_Gvsp_BayerGR10:
        case PixelType_Gvsp_BayerRG10:
        case PixelType_Gvsp_BayerGB10:
        case PixelType_Gvsp_BayerBG10:
        case PixelType_Gvsp_BayerGR12:
        case PixelType_Gvsp_BayerRG12:
        case PixelType_Gvsp_BayerGB12:
        case PixelType_Gvsp_BayerBG12:
        case PixelType_Gvsp_BayerGR16:
        case PixelType_Gvsp_BayerRG16:
        case PixelType_Gvsp_BayerGB16:
        case PixelType_Gvsp_BayerBG16: return sensor_msgs::image_encodings::BAYER_RGGB16;
        default:
            RCLCPP_WARN_ONCE(rclcpp::get_logger("camera_driver"),
                           "未知像素格式 0x%08X，使用默认格式 BAYER_RGGB8", format);
            return sensor_msgs::image_encodings::BAYER_RGGB8;
        }
    }

    CameraControl camera_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_;

    double exposure_;
    double gain_;
    bool trigger_;
    double frame_rate_;
    unsigned int pixel_format_;
    std::string frame_id_;
    std::string serial_number_;

    bool settings_applied_ = false;
    bool last_attempt_valid_ = false;
    rclcpp::Time last_attempt_;
    rclcpp::Time last_fps_time_;
    unsigned int consecutive_failures_ = 0;
    const double reconnect_interval_sec_ = 1.0;
    const unsigned int max_grab_failures_ = 5;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
