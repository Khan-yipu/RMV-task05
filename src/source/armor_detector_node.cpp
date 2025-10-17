// armor_detector_node.cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include <chrono>
#include <vector>
#include <set>
#include <map>
#include <memory>

using namespace std::chrono_literals;

// 灯条结构体
struct LightBar {
    cv::RotatedRect rect;
    float area;
    float length;
    float angle;
    float brightness;
    std::string color;
    float colorScore;
    cv::Point2f topCenter;
    cv::Point2f bottomCenter;
    cv::Point2f verticalCenter;
    bool isMatched;
    int id;

    LightBar() : area(0), length(0), angle(0), brightness(0), colorScore(0),
                isMatched(false), id(0) {}

    void calculateGeometry() {
        cv::Point2f points[4];
        rect.points(points);

        std::vector<cv::Point2f> sortedPoints(points, points + 4);
        std::sort(sortedPoints.begin(), sortedPoints.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });

        topCenter = (sortedPoints[0] + sortedPoints[1]) * 0.5;
        bottomCenter = (sortedPoints[2] + sortedPoints[3]) * 0.5;
        verticalCenter = (topCenter + bottomCenter) * 0.5;
    }
};

// 装甲板结构体
struct ArmorPlate {
    cv::Point2f vertices[4];
    cv::Rect boundingRect;
    std::vector<LightBar> matchedLightBars;
    bool isSimilarParallel;
    bool isSmallArmor;
    float matchScore;
    float rectangleSimilarity;
    int armorId;
    float aspectRatio;
    
    // PNP解算结果
    cv::Mat rvec;
    cv::Mat tvec;
    double distance;
    cv::Point3f position;

    ArmorPlate() : isSimilarParallel(false), isSmallArmor(false),
                  matchScore(0), rectangleSimilarity(0), armorId(0), aspectRatio(0),
                  distance(0) {}
};

class ArmorDetector {
public:
    ArmorDetector() : debug_mode_(true), frame_count_(0) {
        // 初始化相机内参矩阵和畸变系数
        // 从标定文件读取的参数
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            2572.9090799501914, 0.0, 1553.8301019949429,
            0.0, 2591.578784714755, 1053.5377694652548,
            0.0, 0.0, 1.0);
        
        dist_coeffs_ = (cv::Mat_<double>(1, 5) << 
            -0.4315119760105534, 0.24874079388362055, -0.004017564006650052, 
            -0.0016649901347769046, -0.10619591042773283);
        
        // 定义装甲板的3D坐标 (单位：米)
        const double armor_width = 0.135;  // 装甲板宽度 13.5cm
        const double armor_height = 0.056; // 装甲板高度 5.6cm
        armor_points_.push_back(cv::Point3f(-armor_width / 2, -armor_height / 2, 0)); // 左下
        armor_points_.push_back(cv::Point3f(armor_width / 2, -armor_height / 2, 0));  // 右下
        armor_points_.push_back(cv::Point3f(armor_width / 2, armor_height / 2, 0));   // 右上
        armor_points_.push_back(cv::Point3f(-armor_width / 2, armor_height / 2, 0));  // 左上
    }

    void setDebugMode(bool debug) { debug_mode_ = debug; }

    std::vector<ArmorPlate> detect(const cv::Mat& frame) {
        frame_count_++;

        std::vector<ArmorPlate> armors;

        try {
            // 图像预处理
            cv::Mat binary = preprocessImage(frame);

            // 检测灯条
            std::vector<LightBar> lightBars = detectLightBars(binary, frame);

            // 匹配装甲板
            armors = matchArmorPlates(lightBars, frame, binary);

            if (debug_mode_ && !armors.empty()) {
                RCLCPP_DEBUG(rclcpp::get_logger("armor_detector"),
                            "Frame %d: Detected %zu light bars and %zu armors",
                            frame_count_, lightBars.size(), armors.size());
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("armor_detector"),
                        "Error in armor detection: %s", e.what());
        }

        return armors;
    }

    void drawDetectionResult(cv::Mat& frame, const std::vector<ArmorPlate>& armors,
                           const std::vector<LightBar>& lightBars = {}) {
        // 绘制灯条（如果提供）
        for (const auto& bar : lightBars) {
            cv::Scalar color;
            if (bar.color == "red") {
                color = cv::Scalar(0, 0, 255);
            } else if (bar.color == "blue") {
                color = cv::Scalar(255, 0, 0);
            } else if (bar.color == "white") {
                color = cv::Scalar(255, 255, 255);
            } else {
                color = cv::Scalar(0, 255, 0);
            }

            // 绘制灯条竖直中线
            cv::line(frame, bar.topCenter, bar.bottomCenter, color, 2);

            // 标记灯条ID
            std::string info = std::to_string(bar.id);
            cv::putText(frame, info, bar.rect.center, cv::FONT_HERSHEY_SIMPLEX,
                      0.5, cv::Scalar(255, 255, 255), 2);
        }

        // 绘制装甲板，按得分顺序用不同颜色表示优先级
        for (size_t i = 0; i < armors.size(); i++) {
            const auto& armor = armors[i];
            cv::Scalar armorColor;

            // 根据锁定顺序使用不同颜色
            if (i == 0) {
                armorColor = cv::Scalar(0, 255, 0); // 第一顺位：绿色
            } else if (i == 1) {
                armorColor = cv::Scalar(0, 255, 255); // 第二顺位：黄色
            } else if (i == 2) {
                armorColor = cv::Scalar(0, 165, 255); // 第三顺位：橙色
            } else {
                armorColor = cv::Scalar(0, 0, 255); // 其他：红色
            }

            // 绘制装甲板矩形
            for (int j = 0; j < 4; j++) {
                cv::line(frame, armor.vertices[j], armor.vertices[(j+1)%4], armorColor, 3);
            }

            // 显示装甲板信息
            std::string orderText = "Order:" + std::to_string(i + 1);
            std::string scoreText = "Score:" + std::to_string(armor.matchScore);
            std::string ratioText = "Ratio:" + std::to_string(armor.aspectRatio);

            cv::Point textPos(armor.boundingRect.x, armor.boundingRect.y - 10);
            if (textPos.y < 20) textPos.y = armor.boundingRect.y + 20;

            cv::putText(frame, orderText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.4, armorColor, 1);
            cv::putText(frame, scoreText, cv::Point(textPos.x, textPos.y + 15),
                      cv::FONT_HERSHEY_SIMPLEX, 0.4, armorColor, 1);
            cv::putText(frame, ratioText, cv::Point(textPos.x, textPos.y + 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.4, armorColor, 1);
            
            // 绘制PNP解算结果
            if (armor.distance > 0) {
                std::string distanceText = "Dist:" + std::to_string(armor.distance).substr(0, 5) + "m";
                cv::putText(frame, distanceText, cv::Point(textPos.x, textPos.y + 45),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, armorColor, 1);
                
                // 绘制坐标轴可视化位姿
                cv::drawFrameAxes(frame, camera_matrix_, dist_coeffs_, armor.rvec, armor.tvec, 0.1);
            }
        }

        // 绘制统计信息
        std::string stats = "Armors:" + std::to_string(armors.size());
        cv::putText(frame, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  0.8, cv::Scalar(0, 0, 255), 2);
    }

private:
    // PNP解算函数
    bool solvePnPForArmor(ArmorPlate& armor) {
        // 获取图像中的2D角点
        std::vector<cv::Point2f> image_points;
        for (int i = 0; i < 4; i++) {
            image_points.push_back(armor.vertices[i]);
        }
        
        // 对2D点进行排序，确保与3D模型点的顺序一致
        // 按照：左下, 右下, 右上, 左上 的顺序
        std::sort(image_points.begin(), image_points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        
        // 对上下两对点分别按x坐标排序
        if (image_points[0].x > image_points[1].x) {
            std::swap(image_points[0], image_points[1]); // 上方两个点 (左上, 右上)
        }
        if (image_points[2].x < image_points[3].x) {
            std::swap(image_points[2], image_points[3]); // 下方两个点 (右下, 左下)
        }
        
        // 最终顺序调整为：左下, 右下, 右上, 左上
        std::vector<cv::Point2f> sorted_image_points = {
            image_points[3], image_points[2], image_points[1], image_points[0]
        };
        
        // 调用solvePnP
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(armor_points_, sorted_image_points, camera_matrix_, dist_coeffs_, 
                                   rvec, tvec, false, cv::SOLVEPNP_IPPE);
        
        if (success) {
            armor.rvec = rvec.clone();
            armor.tvec = tvec.clone();
            armor.distance = cv::norm(tvec);
            armor.position = cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            
            // 使用INFO级别输出PNP解算结果
            RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
                       "PNP解算成功 - 位置: (%.3f, %.3f, %.3f) m, 距离: %.3f m",
                       armor.position.x, armor.position.y, armor.position.z, armor.distance);
        }
        
        return success;
    }

    cv::Mat preprocessImage(const cv::Mat& input) {
        cv::Mat gray, blurred, binary;

        std::vector<cv::Mat> channels;
        cv::split(input, channels);

        // 增强红色和蓝色通道
        cv::Mat enhanced_gray = 0.5 * channels[2] + 0.4 * channels[0] + 0.1 * channels[1];

        // 中值滤波去噪
        int filterSize = 5;
        if (input.cols < 400 || input.rows < 300) {
            filterSize = 3;
        }
        cv::medianBlur(enhanced_gray, blurred, filterSize);

        // 多种阈值方法结合
        cv::Mat binary_low;
        cv::threshold(blurred, binary_low, 190, 255, cv::THRESH_BINARY);

        cv::Mat binary_high;
        cv::threshold(blurred, binary_high, 240, 255, cv::THRESH_BINARY);

        cv::Mat binary_otsu;
        double otsu_threshold = cv::threshold(blurred, binary_otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

        if (otsu_threshold < 200) {
            cv::threshold(blurred, binary_otsu, std::max(150.0, otsu_threshold), 255, cv::THRESH_BINARY);
        }

        // 合并二值化结果
        cv::Mat binary_combined;
        cv::bitwise_or(binary_otsu, binary_high, binary_combined);
        cv::bitwise_or(binary_combined, binary_low, binary_combined);

        // 形态学操作
        cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
        cv::erode(binary_combined, binary, kernel_erode);

        cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(binary, binary, kernel_dilate);

        return binary;
    }

    std::vector<LightBar> detectLightBars(const cv::Mat& binary, const cv::Mat& original) {
        std::vector<LightBar> lightBars;
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat hsv;
        cv::cvtColor(original, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);
        cv::Mat value_channel = hsv_channels[2];

        static int lightBarId = 0;

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 8) continue;

            cv::RotatedRect rect = cv::minAreaRect(contour);
            float width = rect.size.width;
            float height = rect.size.height;

            if (std::min(width, height) < 1e-5) continue;

            float ratio = std::max(width, height) / std::min(width, height);
            float length = std::max(width, height);

            // 放宽长宽比限制，允许正方形和近似正方形
            if (ratio < 0.8f) {
                continue;
            }

            // 根据形状特征调整筛选条件
            if (ratio <= 1.5f) {
                // 正方形或近似正方形灯条
                if (area < 15.0f) continue;
                if (length < 8.0f) continue;
            }
            else if (ratio <= 3.0f) {
                // 略有扁平的矩形灯条
                if (length < 10.0f) continue;
            }
            else {
                // 细长灯条
                if (length < 15.0f) continue;
                if (ratio > 25.0f) continue;
            }

            // 增加面积与周长比的筛选
            double perimeter = cv::arcLength(contour, true);
            if (perimeter < 1e-5) continue;

            double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
            if (circularity < 0.3f) continue;

            // 计算亮度均值
            cv::Mat mask = cv::Mat::zeros(binary.size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> contour_vec = {contour};
            cv::drawContours(mask, contour_vec, -1, cv::Scalar(255), cv::FILLED);
            cv::Scalar mean_brightness = cv::mean(value_channel, mask);

            // 亮度阈值
            if (mean_brightness[0] < 220) continue;

            // 对于正方形灯条，可以稍微放宽亮度要求
            if (ratio <= 2.0f) {
                if (mean_brightness[0] < 200) continue;
            } else {
                if (mean_brightness[0] < 220) continue;
            }

            // 角度筛选
            float angle = rect.angle;
            bool is_vertical = false;

            if (width < height) {
                angle = 90 - angle;
                is_vertical = true;
            } else {
                angle = -angle;
            }

            // 放宽角度限制
            if (is_vertical) {
                if (abs(angle) < 30 && ratio > 2.0f) continue;
            } else {
                if (abs(angle) < 30 && ratio > 2.0f) continue;
            }

            // 颜色检测
            float colorScore = 0.7;
            std::string color = "unknown";

            // 采样中心点颜色
            int x = std::max(0, std::min(original.cols-1, (int)rect.center.x));
            int y = std::max(0, std::min(original.rows-1, (int)rect.center.y));

            cv::Vec3b pixel = original.at<cv::Vec3b>(y, x);
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];

            if (red > blue + 30 && red > green + 30) {
                color = "red";
                colorScore = 0.8;
            } else if (blue > red + 30 && blue > green + 30) {
                color = "blue";
                colorScore = 0.8;
            } else if (red > 200 && green > 200 && blue > 200) {
                color = "white";
                colorScore = 0.9;
            }

            // 颜色得分阈值
            if (colorScore < 0.3) continue;

            LightBar lightBar;
            lightBar.rect = rect;
            lightBar.area = area;
            lightBar.length = length;
            lightBar.angle = angle;
            lightBar.brightness = mean_brightness[0];
            lightBar.color = color;
            lightBar.colorScore = colorScore;
            lightBar.isMatched = false;
            lightBar.id = lightBarId++;

            lightBar.calculateGeometry();

            lightBars.push_back(lightBar);
        }

        std::sort(lightBars.begin(), lightBars.end(), [](const LightBar& a, const LightBar& b) {
            return a.length > b.length;
        });

        return lightBars;
    }

    float calculateMatchScore(const LightBar& leftBar, const LightBar& rightBar,
                            const std::vector<cv::Point2f>& armorVertices) {
        float score = 0.0f;

        // 计算灯条宽度
        float leftWidth = std::min(leftBar.rect.size.width, leftBar.rect.size.height);
        float rightWidth = std::min(rightBar.rect.size.width, rightBar.rect.size.height);

        // 宽度相似性得分
        float widthDiffRatio = std::abs(leftWidth - rightWidth) / std::max(leftWidth, rightWidth);
        if (widthDiffRatio < 0.5f) {
            score += 0.25f * (1.0f - widthDiffRatio / 0.5f);
        }

        // 长度相似性得分
        float lengthDiffRatio = std::abs(leftBar.length - rightBar.length) / std::max(leftBar.length, rightBar.length);
        if (lengthDiffRatio < 0.4f) {
            score += 0.30f * (1.0f - lengthDiffRatio / 0.4f);
        }

        // 角度关联性得分
        float angleDiff = std::abs(leftBar.angle - rightBar.angle);
        if (angleDiff > 90.0f) {
            angleDiff = 180.0f - angleDiff;
        }

        if (angleDiff < 15.0f) {
            score += 0.20f * (1.0f - angleDiff / 15.0f);
        }

        // 高度对齐得分
        float yAlignment = std::abs(leftBar.verticalCenter.y - rightBar.verticalCenter.y);
        float avgLength = (leftBar.length + rightBar.length) * 0.5;
        float yAlignmentRatio = yAlignment / avgLength;
        if (yAlignmentRatio < 0.5f) {
            score += 0.05f * (1.0f - yAlignmentRatio / 0.5f);
        }

        // 装甲板矩形相似度得分
        float rectangleSimilarity = calculateRectangleSimilarity(armorVertices);
        score += 0.20f * rectangleSimilarity;

        return score;
    }

    float calculateRectangleSimilarity(const std::vector<cv::Point2f>& vertices) {
        if (vertices.size() != 4) return 0.0f;

        // 计算四边形的面积
        std::vector<cv::Point2f> hull;
        cv::convexHull(vertices, hull);
        float area = cv::contourArea(hull);

        // 计算最小外接矩形面积
        cv::RotatedRect minRect = cv::minAreaRect(vertices);
        float rectArea = minRect.size.width * minRect.size.height;

        if (rectArea < 1e-5) return 0.0f;

        // 面积比越接近1，越接近矩形
        float areaRatio = area / rectArea;

        // 计算角度接近直角的程度
        std::vector<float> angles;
        for (int i = 0; i < 4; i++) {
            cv::Point2f v1 = vertices[i] - vertices[(i+3)%4];
            cv::Point2f v2 = vertices[(i+1)%4] - vertices[i];
            float dot = v1.x * v2.x + v1.y * v2.y;
            float len1 = std::sqrt(v1.x*v1.x + v1.y*v1.y);
            float len2 = std::sqrt(v2.x*v2.x + v2.y*v2.y);
            if (len1 < 1e-5 || len2 < 1e-5) continue;
            float angle = std::acos(dot / (len1 * len2)) * 180 / CV_PI;
            angles.push_back(angle);
        }

        float angleScore = 0.0f;
        if (!angles.empty()) {
            float angleDiffSum = 0.0f;
            for (float angle : angles) {
                float diff = std::abs(angle - 90.0f);
                angleDiffSum += std::max(0.0f, 1.0f - diff / 45.0f);
            }
            angleScore = angleDiffSum / angles.size();
        }

        // 综合得分
        float similarity = areaRatio * 0.6f + angleScore * 0.4f;
        return similarity;
    }

    float calculateAspectRatioFromMidlines(const LightBar& leftBar, const LightBar& rightBar) {
        float width = cv::norm(leftBar.verticalCenter - rightBar.verticalCenter);
        float height = (leftBar.length + rightBar.length) * 0.5f;

        if (height < 1e-5) return 0.0f;
        return width / height;
    }

    ArmorPlate createArmorFromLightBars(const LightBar& leftBar, const LightBar& rightBar,
                                       const cv::Mat& original, bool isSmall) {
        ArmorPlate armor;
        armor.isSmallArmor = isSmall;

        // 确定左右灯条
        const LightBar* left = &leftBar;
        const LightBar* right = &rightBar;
        if (leftBar.verticalCenter.x > rightBar.verticalCenter.x) {
            std::swap(left, right);
        }

        // 基于中线端点构建装甲板四边形
        armor.vertices[0] = left->topCenter;
        armor.vertices[1] = right->topCenter;
        armor.vertices[2] = right->bottomCenter;
        armor.vertices[3] = left->bottomCenter;

        std::vector<cv::Point2f> armorPointsVec;
        for (int i = 0; i < 4; i++) {
            armorPointsVec.push_back(armor.vertices[i]);
        }

        armor.boundingRect = cv::boundingRect(armorPointsVec);
        armor.boundingRect &= cv::Rect(0, 0, original.cols, original.rows);

        // 计算基于中线端点的长宽比
        armor.aspectRatio = calculateAspectRatioFromMidlines(*left, *right);

        return armor;
    }

    bool validateArmor(const ArmorPlate& armor, const cv::Mat& original, const cv::Mat& binary) {
        // 检查边界矩形有效性
        if (armor.boundingRect.width < 3 || armor.boundingRect.height < 3) {
            return false;
        }

        // 检查边界矩形是否在图像范围内
        if (armor.boundingRect.x < 0 || armor.boundingRect.y < 0 ||
            armor.boundingRect.x + armor.boundingRect.width > original.cols ||
            armor.boundingRect.y + armor.boundingRect.height > original.rows) {
            return false;
        }

        // 检查matchedLightBars是否有效
        if (armor.matchedLightBars.size() < 2) {
            return false;
        }

        // 长宽比验证
        if (armor.aspectRatio < 0.7f || armor.aspectRatio > 5.0f) {
            return false;
        }

        return true;
    }

    std::vector<ArmorPlate> matchArmorPlates(std::vector<LightBar>& lightBars,
                                            const cv::Mat& original, const cv::Mat& binary) {
        std::vector<ArmorPlate> armors;

        if (lightBars.size() < 2) return armors;

        // 按x坐标排序
        std::sort(lightBars.begin(), lightBars.end(), [](const LightBar& a, const LightBar& b) {
            return a.verticalCenter.x < b.verticalCenter.x;
        });

        // 存储所有可能的匹配对
        struct MatchCandidate {
            int leftIndex;
            int rightIndex;
            float score;
            ArmorPlate armor;
            bool isValid;
        };

        std::vector<MatchCandidate> allCandidates;
        static int armorIdCounter = 0;

        // 计算所有灯条组合的得分
        for (size_t i = 0; i < lightBars.size(); i++) {
            for (size_t j = i + 1; j < lightBars.size(); j++) {
                LightBar& leftBar = lightBars[i];
                LightBar& rightBar = lightBars[j];

                // 创建临时装甲板
                bool isSmall = (leftBar.length < 30 && rightBar.length < 30);
                ArmorPlate tempArmor = createArmorFromLightBars(leftBar, rightBar, original, isSmall);

                // 构建装甲板顶点
                std::vector<cv::Point2f> armorVertices;
                for (int k = 0; k < 4; k++) {
                    armorVertices.push_back(tempArmor.vertices[k]);
                }

                float matchScore = calculateMatchScore(leftBar, rightBar, armorVertices);

                // 创建完整的装甲板对象
                ArmorPlate armor = tempArmor;
                armor.matchScore = matchScore;
                armor.isSimilarParallel = (std::abs(leftBar.angle - rightBar.angle) < 15.0f);
                armor.matchedLightBars = {leftBar, rightBar};
                armor.rectangleSimilarity = calculateRectangleSimilarity(armorVertices);
                armor.armorId = armorIdCounter++;

                MatchCandidate candidate;
                candidate.leftIndex = i;
                candidate.rightIndex = j;
                candidate.score = matchScore;
                candidate.armor = armor;
                candidate.isValid = (matchScore > 0.75f) && validateArmor(armor, original, binary);

                allCandidates.push_back(candidate);
            }
        }

        // 筛选有效候选并按得分排序
        std::vector<MatchCandidate> potentialMatches;
        for (const auto& candidate : allCandidates) {
            if (candidate.isValid) {
                potentialMatches.push_back(candidate);
            }
        }

        std::sort(potentialMatches.begin(), potentialMatches.end(),
                 [](const MatchCandidate& a, const MatchCandidate& b) {
                     return a.score > b.score;
                 });

        // 选择装甲板，避免重复使用灯条
        std::set<int> usedLightBars;

        for (const auto& candidate : potentialMatches) {
            int i = candidate.leftIndex;
            int j = candidate.rightIndex;

            LightBar& leftBar = lightBars[i];
            LightBar& rightBar = lightBars[j];

            // 检查灯条是否已经被使用
            if (usedLightBars.find(leftBar.id) != usedLightBars.end() ||
                usedLightBars.find(rightBar.id) != usedLightBars.end()) {
                continue;
            }

            // 创建装甲板
        ArmorPlate armor = candidate.armor;
        
        // 对检测到的装甲板进行PNP解算
        if (solvePnPForArmor(armor)) {
            armors.push_back(armor);
        }

        usedLightBars.insert(leftBar.id);
        usedLightBars.insert(rightBar.id);
        }

        // 按得分对最终装甲板排序
        std::sort(armors.begin(), armors.end(), [](const ArmorPlate& a, const ArmorPlate& b) {
            return a.matchScore > b.matchScore;
        });

        return armors;
    }

    bool debug_mode_;
    int frame_count_;
    
    // PNP解算相关
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> armor_points_;
};

class ArmorDetectorNode : public rclcpp::Node {
public:
    ArmorDetectorNode() : Node("armor_detector") {
        // 声明参数
        this->declare_parameter<bool>("debug_mode", true);
        this->declare_parameter<double>("process_interval", 0.0);
        this->declare_parameter<std::string>("image_topic", "/image_raw");
        this->declare_parameter<std::string>("result_topic", "/armor_detection_result");
        this->declare_parameter<bool>("enable_statistics", true);
        this->declare_parameter<int>("statistics_interval", 30);

        // 获取参数
        bool debug_mode = this->get_parameter("debug_mode").as_bool();
        double process_interval = this->get_parameter("process_interval").as_double();
        std::string image_topic = this->get_parameter("image_topic").as_string();
        std::string result_topic = this->get_parameter("result_topic").as_string();
        bool enable_statistics = this->get_parameter("enable_statistics").as_bool();
        int statistics_interval = this->get_parameter("statistics_interval").as_int();

        // 创建装甲板检测器
        detector_ = std::make_unique<ArmorDetector>();
        detector_->setDebugMode(debug_mode);

        // 创建订阅者和发布者
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic,
            rclcpp::QoS(10).best_effort(),
            std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));

        result_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            result_topic, 10);

        // 初始化统计信息
        total_frames_ = 0;
        detected_frames_ = 0;
        average_process_time_ = 0.0;
        enable_statistics_ = enable_statistics;
        statistics_interval_ = statistics_interval;

        RCLCPP_INFO(this->get_logger(), "Armor Detector Node Started");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", image_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: %s", result_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Debug mode: %s", debug_mode ? "enabled" : "disabled");
        RCLCPP_INFO(this->get_logger(), "Statistics: %s", enable_statistics ? "enabled" : "disabled");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            // 转换为OpenCV格式
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            // 复制图像用于处理
            cv::Mat frame = cv_ptr->image.clone();

            // 处理图像 - 检测装甲板
            std::vector<ArmorPlate> armors = detector_->detect(frame);

            // 绘制检测结果
            if (!armors.empty()) {
                detector_->drawDetectionResult(frame, armors);
                detected_frames_++;

                RCLCPP_DEBUG(this->get_logger(), "Detected %zu armors in frame", armors.size());
            }

            // 发布处理结果
            if (result_publisher_->get_subscription_count() > 0) {
                auto result_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
                result_publisher_->publish(*result_msg);
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in image_callback: %s", e.what());
            return;
        }

        // 计算处理时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double process_time = duration.count();

        // 更新统计信息
        total_frames_++;

        if (enable_statistics_) {
            // 更新平均处理时间
            average_process_time_ = (average_process_time_ * (total_frames_ - 1) + process_time) / total_frames_;

            // 定期输出统计信息
            if (total_frames_ % statistics_interval_ == 0) {
                double detection_rate = (static_cast<double>(detected_frames_) / total_frames_) * 100.0;

                RCLCPP_INFO(this->get_logger(),
                           "Processing Statistics:");
                RCLCPP_INFO(this->get_logger(),
                           "  Frames processed: %d", total_frames_);
                RCLCPP_INFO(this->get_logger(),
                           "  Detection rate: %.1f%% (%d/%d)",
                           detection_rate, detected_frames_, total_frames_);
                RCLCPP_INFO(this->get_logger(),
                           "  Average processing time: %.2f ms", average_process_time_);
                RCLCPP_INFO(this->get_logger(),
                           "  Current FPS: %.1f", 1000.0 / average_process_time_);
            }
        }

        // 调试信息 - 每帧处理时间
        RCLCPP_DEBUG(this->get_logger(), "Frame processed in: %.2f ms", process_time);
    }

    std::unique_ptr<ArmorDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_publisher_;

    int total_frames_;
    int detected_frames_;
    double average_process_time_;
    bool enable_statistics_;
    int statistics_interval_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmorDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
