#include <ros/ros.h>
#include <tf/tf.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/Marker.h>
#include <vector>
#include <cmath>
#include <mutex>
#include <random>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense> 
#include <pcl/kdtree/kdtree_flann.h>


#define MAX_THRUST (double)(((109 * 10) * (109 * 10) * 5.84e-06) * 4.0)

namespace MPPI {
    const int K = 300;                  // Number of samples
    const int N = 20;                   // Prediction horizon
    const double dt = 0.1;               // Sampling time
    const double lambda =1e-6;           // Temperature for cost weighting
    const double control_noise[4] = {3.0, 0.01*0.63, 0.01*0.39, 0.01*0.18};    // Noise level for control samples
    const double H_inv_G = 1.0;         // Simplification;
    
    const double dist_to_goal_weight = 25;
    const double direction_weight = 25;
    const double dist_weight = 20;
    const double obs_weight = 1e10;
    const double control_weight = 500;
    const double safe_distance_to_obs = 0.5;
}

namespace QuadParams {
    const double mass = 1.527;
    const double gravConst = 9.8066;
    constexpr double ixx = 0.029125;
    constexpr double iyy = 0.029125;
    constexpr double izz = 0.055225;

    constexpr double inv_ixx = 1 / ixx;
    constexpr double inv_iyy = 1 / iyy;
    constexpr double inv_izz = 1 / izz;

    constexpr double c_zy = (izz - iyy);
    constexpr double c_xz = (ixx - izz);
    constexpr double c_yx = (iyy - ixx);
}

struct Controls {
    double ct;
    double tx, ty, tz;
};

class MPPIController {
    public:
        MPPIController(ros::NodeHandle nh, const geometry_msgs::PoseStamped& start_point, const geometry_msgs::PoseStamped& goal_point)
            : nh(nh), start_point(start_point), goal_point(goal_point) 
        {
            trajectory_marker_pub = nh.advertise<visualization_msgs::Marker>("trajectory_marker", 10);
            selected_marker_pub = nh.advertise<visualization_msgs::Marker>("selected_marker", 10);

            // Proper initialization inside the constructor
            sampled_position.resize(MPPI::K);
            for (int i = 0; i < MPPI::K; ++i) {
                sampled_position[i].resize(MPPI::N);
            }
            selected_position.resize(MPPI::N + 1);
            selected_states.resize(MPPI::N + 1);

            ROS_INFO("[MPPI] MPPI control class has been successfully created!");
        }

        double compute_cost(const nav_msgs::Odometry& states, Controls& controls, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree) {
            double cost = 0.0;

            double dist_to_goal_cost = compute_dist_to_goal_cost(states);
            double direction_cost = compute_direction_cost(states);
            double obstacle_cost = compute_obstacle_cost(states, kdtree);
            double control_cost = compute_control_cost(controls);
            double hovering_cost = compute_hovering_cost(states);

            // cost = (MPPI::dist_to_goal_weight * dist_to_goal_cost) + (MPPI::direction_weight * direction_cost) + (MPPI::obs_weight * obstacle_cost) + (MPPI::control_weight * control_cost);
            cost = hovering_cost + control_cost;

            return cost;
        }

        double compute_cost(const nav_msgs::Odometry& states, Controls& controls) {
            double cost = 0.0;

            double dist_to_goal_cost = compute_dist_to_goal_cost(states);
            double direction_cost = compute_direction_cost(states);
            double control_cost = compute_control_cost(controls);
            double hovering_cost = compute_hovering_cost(states);

            // cost = (MPPI::dist_to_goal_weight * dist_to_goal_cost) + (MPPI::direction_weight * direction_cost) + (MPPI::control_weight * control_cost);
            cost = hovering_cost + MPPI::control_weight * control_cost;

            return cost;
        }

        double compute_hovering_cost(const nav_msgs::Odometry& states) {

            // Weights
            double Cp = 400.0;
            double Cv = 40.0;
            double Cq = 20.0;
            double Cw = 20.0;

            // Desried states
            double pos_x_des = goal_point.pose.position.x;
            double pos_y_des = goal_point.pose.position.y;
            double pos_z_des = goal_point.pose.position.z;

            double vel_x_des = 0.0;
            double vel_y_des = 0.0;
            double vel_z_des = 0.0;

            double qw_des = 1.0;
            double qx_des = 0.0;
            double qy_des = 0.0;
            double qz_des = 0.0;

            double wx_des = 0.0;
            double wy_des = 0.0;
            double wz_des = 0.0;

            // Current states
            double px = states.pose.pose.position.x;
            double py = states.pose.pose.position.y;
            double pz = states.pose.pose.position.z;

            double vx = states.twist.twist.linear.x;
            double vy = states.twist.twist.linear.y;
            double vz = states.twist.twist.linear.z;

            double qw = states.pose.pose.orientation.w;
            double qx = states.pose.pose.orientation.x;
            double qy = states.pose.pose.orientation.y;
            double qz = states.pose.pose.orientation.z;

            double wx = states.twist.twist.angular.x;
            double wy = states.twist.twist.angular.y;
            double wz = states.twist.twist.angular.z;

            // Compute position cost
            double pos_cost = std::pow(pos_x_des - px, 2) + std::pow(pos_y_des - py, 2) + std::pow(pos_z_des - pz, 2);

            // Compute velocity cost
            double vel_cost = std::pow(vel_x_des - vx, 2) + std::pow(vel_z_des - vz, 2) + std::pow(vel_z_des - vz, 2); 

            // Compute attitude cost
            double inner_product = qw_des * qw + qx_des * qx + qy_des * qy + qz_des * qz;
            double attitude_cost = 1.0 - std::pow(std::abs(inner_product), 2);

            // Compute angular rate cost
            double omega_cost = std::pow(wx_des - wx, 2) + std::pow(wy_des - wy, 2) + std::pow(wz_des - wz, 2);

            return (Cp*pos_cost) + (Cv*vel_cost) + (Cq*attitude_cost) + (Cw*omega_cost);
        }

        double compute_dist_to_goal_cost(const nav_msgs::Odometry& states) {
            double px = states.pose.pose.position.x;
            double py = states.pose.pose.position.y;
            double pz = states.pose.pose.position.z;

            double xg = goal_point.pose.position.x;
            double yg = goal_point.pose.position.y;
            double zg = goal_point.pose.position.z;

            double dist = std::pow(px - xg, 2) + std::pow(py - yg, 2) + std::pow(pz - zg, 2);
            return dist;
        }

        double compute_dist_to_global_path_cost(const nav_msgs::Odometry& states) {

            // Use the same distance computation logic as previously discussed.
            double px = states.pose.pose.position.x;
            double py = states.pose.pose.position.y;
            double pz = states.pose.pose.position.z;

            double xs = start_point.pose.position.x;
            double ys = start_point.pose.position.y;
            double zs = start_point.pose.position.z;
            double xg = goal_point.pose.position.x;
            double yg = goal_point.pose.position.y;
            double zg = goal_point.pose.position.z;

            double dx = xg - xs;
            double dy = yg - ys;
            double dz = zg - zs;

            double projLen = ((px - xs) * dx + (py - ys) * dy + (pz - zs) * dz) / (dx * dx + dy * dy + dz * dz);

            // If projLen is negative, give a large penalty (trajectory is going in the opposite direction)
            if (projLen < 0) {
                return 1e100;  // Super large penalty
            }

            double xc = xs + projLen * dx;
            double yc = ys + projLen * dy;
            double zc = zs + projLen * dz;

            double dist = std::sqrt(std::pow(px - xc, 2) + std::pow(py - yc, 2) + std::pow(pz - zc, 2));
            return dist;
        }

        double compute_direction_cost(const nav_msgs::Odometry& states) {
            // Vector from current position to goal
            double dx = goal_point.pose.position.x - states.pose.pose.position.x;
            double dy = goal_point.pose.position.y - states.pose.pose.position.y;
            double dz = goal_point.pose.position.z - states.pose.pose.position.z;

            double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (norm < 1e-6) return 0.0;

            // Normalize direction vector to goal
            dx /= norm;
            dy /= norm;
            dz /= norm;

            // Current velocity vector
            double vx = states.twist.twist.linear.x;
            double vy = states.twist.twist.linear.y;
            double vz = states.twist.twist.linear.z;

            double velocity_magnitude = std::sqrt(vx * vx + vy * vy + vz * vz);
            if (velocity_magnitude < 1e-6) return 0.0;

            // Normalize velocity vector
            vx /= velocity_magnitude;
            vy /= velocity_magnitude;
            vz /= velocity_magnitude;

            // Compute cost as the negative of the dot product
            // Higher cost when the drone is not moving towards the goal
            double direction_cost = 1.0 - (dx * vx + dy * vy + dz * vz);
            return direction_cost;
        }

        double compute_obstacle_cost(const nav_msgs::Odometry& states, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree) {
            double obstacle_cost = 0.0;
            double max_check_radius = MPPI::safe_distance_to_obs;

            double px = states.pose.pose.position.x;
            double py = states.pose.pose.position.y;
            double pz = states.pose.pose.position.z;

            pcl::PointXYZ search_point;
            search_point.x = px;
            search_point.y = py;
            search_point.z = pz;

            std::vector<int> point_idx_radius_search;
            std::vector<float> point_radius_squared_distance;

            double safe_dist_sq = MPPI::safe_distance_to_obs * MPPI::safe_distance_to_obs;

            // Perform nearest radius search
            if (kdtree.radiusSearch(search_point, max_check_radius, point_idx_radius_search, point_radius_squared_distance) > 0) {
                for (size_t i = 0; i < point_idx_radius_search.size(); ++i) {
                    double distance_to_obstacle_sq = point_radius_squared_distance[i];

                    // If the distance is less than the safe distance, return 1
                    if (distance_to_obstacle_sq < safe_dist_sq) {
                        return 1.0;
                    }
                }
            }

            return 0.0;
        }

        double compute_control_cost(const Controls& controls) {
            return std::pow(controls.ct - QuadParams::mass*QuadParams::gravConst, 2) + std::pow(controls.tx, 2) + std::pow(controls.ty, 2) + std::pow(controls.tz, 2);
        }

        void mppi_optimization(const nav_msgs::Odometry& states, std::vector<Controls>& control_sequence, const pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles) {
            // Set up the KD-tree for nearest neighbor search
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

            bool valid_obstacle = !obstacles->empty();
            if (valid_obstacle) {
                kdtree.setInputCloud(obstacles);
            } else {
                // ROS_INFO("[MPPI] Point cloud is empty. No obstacles detected.");
            }

            // Containers for storing control updates and costs
            nav_msgs::Odometry sampled_states = states;
            std::vector<Controls> control_update(MPPI::N, {0.0, 0.0, 0.0, 0.0});
            std::vector<double> cost_weights(MPPI::K, 0.0);
            std::vector<std::vector<Controls>> noisy_controls(MPPI::K, std::vector<Controls>(MPPI::N));
            double dt_sqrt = std::sqrt(MPPI::dt);

            // Loop through K samples
            for (int k = 0; k < MPPI::K; ++k) {
                sampled_states = states;
                double total_cost = 0.0;

                // Generate a noisy trajectory and accumulate cost
                for (int i = 0; i < MPPI::N; ++i) {
                    noisy_controls[k][i].ct = control_sequence[i].ct + thrust_distribution(noise_generator);
                    noisy_controls[k][i].tx = control_sequence[i].tx + torque_x_distribution(noise_generator);
                    noisy_controls[k][i].ty = control_sequence[i].ty + torque_y_distribution(noise_generator);
                    noisy_controls[k][i].tz = control_sequence[i].tz + torque_z_distribution(noise_generator);

                    updateOmega(sampled_states, noisy_controls[k][i]);
                    updateQuaternion(sampled_states);
                    updateVelocity(sampled_states, noisy_controls[k][i]);
                    updatePosition(sampled_states);

                    sampled_position[k][i].point.x = sampled_states.pose.pose.position.x;
                    sampled_position[k][i].point.y = sampled_states.pose.pose.position.y;
                    sampled_position[k][i].point.z = sampled_states.pose.pose.position.z;

                    // Only compute obstacle cost if we have valid obstacles
                    if (valid_obstacle) {
                        total_cost += compute_cost(sampled_states, noisy_controls[k][i], kdtree);
                    } else {
                        // Skip obstacle cost, only compute goal-related costs
                        total_cost += compute_cost(sampled_states, noisy_controls[k][i]);
                    }
                }

                // Calculate the exponential weighting for this sample based on its total cost
                cost_weights[k] = total_cost / MPPI::lambda;
            }
            double min_weight = *std::min_element(cost_weights.begin(), cost_weights.end());
            for (int k = 0; k < MPPI::K; ++k) {
                cost_weights[k] = std::exp(min_weight - cost_weights[k]);
            }

            // Normalize the cost weights for stability
            double weight_sum = std::accumulate(cost_weights.begin(), cost_weights.end(), 0.0);
            for (int k = 0; k < MPPI::K; ++k) {
                cost_weights[k] /= weight_sum;
            }

            // Calculate weighted control update for each time step
            for (int i = 0; i < MPPI::N; ++i) {
                double ct_sum = 0.0;
                double tx_sum = 0.0;
                double ty_sum = 0.0;
                double tz_sum = 0.0;

                for (int k = 0; k < MPPI::K; ++k) {
                    // Scale the noise by the time step and temperature
                    double noise_ct = (noisy_controls[k][i].ct - control_sequence[i].ct) * dt_sqrt;
                    double noise_tx = (noisy_controls[k][i].tx - control_sequence[i].tx) * dt_sqrt;
                    double noise_ty = (noisy_controls[k][i].ty - control_sequence[i].ty) * dt_sqrt;
                    double noise_tz = (noisy_controls[k][i].tz - control_sequence[i].tz) * dt_sqrt;

                    // Accumulate the weighted control adjustments
                    ct_sum += cost_weights[k] * noise_ct;
                    tx_sum += cost_weights[k] * noise_tx;
                    ty_sum += cost_weights[k] * noise_ty;
                    tz_sum += cost_weights[k] * noise_tz;
                }

                // Normalize and apply H_inv_G
                if (weight_sum != 0.0) {
                    control_update[i].ct = MPPI::H_inv_G * (ct_sum / weight_sum);
                    control_update[i].tx = MPPI::H_inv_G * (tx_sum / weight_sum);
                    control_update[i].ty = MPPI::H_inv_G * (ty_sum / weight_sum);
                    control_update[i].tz = MPPI::H_inv_G * (tz_sum / weight_sum);
                }
                else {
                    control_update[i].ct = 0.0;
                    control_update[i].tx = 0.0;
                    control_update[i].ty = 0.0;
                    control_update[i].tz = 0.0;
                }
            }
            
            // Update control sequence incrementally
            selected_position[0].point.x = states.pose.pose.position.x;
            selected_position[0].point.y = states.pose.pose.position.y;
            selected_position[0].point.z = states.pose.pose.position.z;

            // Update optiaml states
            selected_states.clear();
            selected_states.push_back(states);

            // Optimal states computation
            sampled_states = states;

            for (int i = 0; i < MPPI::N; ++i) {
                control_sequence[i].ct += control_update[i].ct;
                control_sequence[i].tx += control_update[i].tx;
                control_sequence[i].ty += control_update[i].ty;
                control_sequence[i].tz += control_update[i].tz;

                updateOmega(sampled_states, control_sequence[i]);
                updateQuaternion(sampled_states);
                updateVelocity(sampled_states, control_sequence[i]);
                updatePosition(sampled_states);

                selected_states.push_back(sampled_states);

                selected_position[i+1].point.x = sampled_states.pose.pose.position.x;
                selected_position[i+1].point.y = sampled_states.pose.pose.position.y;
                selected_position[i+1].point.z = sampled_states.pose.pose.position.z;
            }

            sampled_position_publish();
            selected_position_publish();
        }

        inline void updateOmega(nav_msgs::Odometry& states, const Controls& controls) {
            double temp_wx = states.twist.twist.angular.x;
            double temp_wy = states.twist.twist.angular.y;
            double temp_wz = states.twist.twist.angular.z;

            // Update state based on noisy control
            states.twist.twist.angular.x += MPPI::dt * QuadParams::inv_ixx * (QuadParams::c_zy * temp_wy * temp_wz + controls.tx);
            states.twist.twist.angular.y += MPPI::dt * QuadParams::inv_iyy * (QuadParams::c_xz * temp_wx * temp_wz + controls.ty);   
            states.twist.twist.angular.z += MPPI::dt * QuadParams::inv_izz * (QuadParams::c_yx * temp_wy * temp_wx + controls.tz);
        }

        inline void updateQuaternion(nav_msgs::Odometry& states) {
            // References to quaternion components for readability
            double& qw = states.pose.pose.orientation.w;
            double& qx = states.pose.pose.orientation.x;
            double& qy = states.pose.pose.orientation.y;
            double& qz = states.pose.pose.orientation.z;

            // References to angular velocity components
            const double& wx = states.twist.twist.angular.x;
            const double& wy = states.twist.twist.angular.y;
            const double& wz = states.twist.twist.angular.z;

            // Store original quaternion values for updates
            double qw_temp = qw;
            double qx_temp = qx;
            double qy_temp = qy;
            double qz_temp = qz;

            // Update quaternion components in place
            qw += MPPI::dt * 0.5 * (-qx_temp * wx - qy_temp * wy - qz_temp * wz);
            qx += MPPI::dt * 0.5 * (qw_temp * wx + qy_temp * wz - qz_temp * wy);
            qy += MPPI::dt * 0.5 * (qw_temp * wy - qx_temp * wz + qz_temp * wx);
            qz += MPPI::dt * 0.5 * (qw_temp * wz + qx_temp * wy - qy_temp * wx);

            // Normalize quaternion in place
            const double norm_inv = 1.0 / std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw *= norm_inv;
            qx *= norm_inv;
            qy *= norm_inv;
            qz *= norm_inv;
        }

        inline void updateVelocity(nav_msgs::Odometry& states, const Controls& controls) {
            // References to quaternion components for readability
            double& qw = states.pose.pose.orientation.w;
            double& qx = states.pose.pose.orientation.x;
            double& qy = states.pose.pose.orientation.y;
            double& qz = states.pose.pose.orientation.z;

            double normalized_thrust = controls.ct / QuadParams::mass;
            states.twist.twist.linear.x += MPPI::dt * 2.0 * (qw * qy - qz * qx) * normalized_thrust;
            states.twist.twist.linear.y += MPPI::dt * 2.0 * (qw * qz + qx * qy) * normalized_thrust;
            states.twist.twist.linear.z += MPPI::dt * ((qw * qw - qx * qx - qy * qy + qz * qz) * normalized_thrust - QuadParams::gravConst); 
        }

        inline void updatePosition(nav_msgs::Odometry& states) {
            states.pose.pose.position.x += MPPI::dt * states.twist.twist.linear.x;
            states.pose.pose.position.y += MPPI::dt * states.twist.twist.linear.y;
            states.pose.pose.position.z += MPPI::dt * states.twist.twist.linear.z;
        }

        std::vector<nav_msgs::Odometry>& get_optimal_states() {
            return selected_states;
        }

        void sampled_position_publish() {
            visualization_msgs::Marker line_strip;
            line_strip.header.frame_id = "map";  // Set the appropriate frame
            line_strip.header.stamp = ros::Time::now();
            line_strip.ns = "trajectory";
            line_strip.id = 0;
            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            line_strip.action = visualization_msgs::Marker::ADD;
            line_strip.scale.x = 0.05;  // Line width
            line_strip.color.r = 0.0;
            line_strip.color.g = 1.0;
            line_strip.color.b = 0.0;
            line_strip.color.a = 0.1;  // Fully opaque

            // Add all sampled points to the line_strip marker
            for (int k = 0; k < MPPI::K; ++k) {
                for (int i = 0; i < MPPI::N; ++i) {
                    geometry_msgs::Point p;
                    p.x = sampled_position[k][i].point.x;
                    p.y = sampled_position[k][i].point.y;
                    p.z = sampled_position[k][i].point.z;
                    line_strip.points.push_back(p);
                }
            }

            // Publish the line strip marker
            trajectory_marker_pub.publish(line_strip);
        }

        void selected_position_publish() {
            visualization_msgs::Marker line_strip;
            line_strip.header.frame_id = "map";  // Set the appropriate frame
            line_strip.header.stamp = ros::Time::now();
            line_strip.ns = "selected";
            line_strip.id = 0;
            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            line_strip.action = visualization_msgs::Marker::ADD;
            line_strip.scale.x = 0.20;  // Line width
            line_strip.color.r = 1.0;
            line_strip.color.g = 0.0;
            line_strip.color.b = 0.0;
            line_strip.color.a = 1.0;  // Fully opaque

            // Add all sampled points to the line_strip marker
            for (int i = 0; i < MPPI::N; ++i) {
                geometry_msgs::Point p;
                p.x = selected_position[i].point.x;
                p.y = selected_position[i].point.y;
                p.z = selected_position[i].point.z;
                line_strip.points.push_back(p);
            }

            // Publish the line strip marker
            selected_marker_pub.publish(line_strip);
        }


    private:
        ros::NodeHandle nh;

        geometry_msgs::PoseStamped start_point;
        geometry_msgs::PoseStamped goal_point;

        // Random noise generator
        std::default_random_engine noise_generator;
        std::normal_distribution<double> thrust_distribution{0.0, MPPI::control_noise[0]};
        std::normal_distribution<double> torque_x_distribution{0.0, MPPI::control_noise[1]};
        std::normal_distribution<double> torque_y_distribution{0.0, MPPI::control_noise[2]};
        std::normal_distribution<double> torque_z_distribution{0.0, MPPI::control_noise[3]};

        // Visualization of computed positions
        std::vector<std::vector<geometry_msgs::PointStamped>> sampled_position;
        std::vector<geometry_msgs::PointStamped> selected_position;

        // Optimal states
        std::vector<nav_msgs::Odometry> selected_states;

        ros::Publisher trajectory_marker_pub;
        ros::Publisher selected_marker_pub;
};

class QuadStatesScriber {
    public:
        QuadStatesScriber(ros::NodeHandle nh) : nh(nh), data_received(false) {
            quad_states_sub = nh.subscribe("/mavros/global_position/local", 10, &QuadStatesScriber::quadStateCallback, this);
            quad_imu_sub = nh.subscribe("/mavros/imu/data", 10, &QuadStatesScriber::quadImuCallback, this);
            ROS_INFO("[MPPI] Quadrotor state scriber has been successfully created!");
        }

        void quadStateCallback(const nav_msgs::Odometry& msg) {
            std::lock_guard<std::mutex> lock(mutex_);
            quad_states = msg;
            data_received = true;  // Indicate that data has been received
        }

        void quadImuCallback(const sensor_msgs::Imu& msg) {
            std::lock_guard<std::mutex> lock(mutex_imu);
            quad_imu = msg;
        }

        nav_msgs::Odometry getQuadStates() {
            std::lock_guard<std::mutex> lock(mutex_);
            return quad_states;
        }

        sensor_msgs::Imu getQuadImu() {
            std::lock_guard<std::mutex> lock(mutex_imu);
            return quad_imu;
        }

        bool isDataAvailable() {
            std::lock_guard<std::mutex> lock(mutex_);
            return data_received;
        }

    private:
        ros::NodeHandle nh;
        ros::Subscriber quad_states_sub;
        ros::Subscriber quad_imu_sub;
        nav_msgs::Odometry quad_states;
        sensor_msgs::Imu quad_imu;
        std::mutex mutex_;
        std::mutex mutex_imu;
        bool data_received;
};

class SensorDataManager {
    public:
        SensorDataManager(ros::NodeHandle nh, QuadStatesScriber& quad_states_subscriber)
            : nh(nh), quad_states_subscriber(quad_states_subscriber), data_received(false), octree(0.2) // nh("~"): node handle with private namespace for the node, OctoMap with 5 cm resolution
        {
            // Subscribe to depth point cloud topics
            point_cloud_sub = nh.subscribe("/iris_depth_camera/camera/depth/points", 1, &SensorDataManager::pointCloudCallback, this);

            // Initialize the point cloud.
            current_point_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

            ROS_INFO("[MPPI] Sensor data manager class has been successfully created!");

            // Sensor offset position
            sensor_offset.pose.position.x = 0.1;
            sensor_offset.pose.position.y = 0.0;
            sensor_offset.pose.position.z = 0.0;

            point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformedPT", 1);
            octomap_pub = nh.advertise<octomap_msgs::Octomap>("/octomap", 1);
        }

        void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {

            octree.clear();
            
            nav_msgs::Odometry quad_states = quad_states_subscriber.getQuadStates();

            std::lock_guard<std::mutex> lock(mutex_);

            // Convert sensor_msgs::PointCloud2 to PCL PointCloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);

            // Remove NaNs from the cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

            // Apply VoxelGrid downsampling
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(cloud_filtered);
            voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f); // Set the voxel grid size, adjust values as needed
            voxel_filter.filter(*cloud_filtered);

            // Transformation based on quadrotor position and orientation
            double tx = quad_states.pose.pose.position.x;
            double ty = quad_states.pose.pose.position.y;
            double tz = quad_states.pose.pose.position.z;

            tf::Quaternion q(
                quad_states.pose.pose.orientation.x,
                quad_states.pose.pose.orientation.y,
                quad_states.pose.pose.orientation.z,
                quad_states.pose.pose.orientation.w
            );
            tf::Matrix3x3 rotation_matrix(q);

            // Define the fixed rotation matrix from point cloud frame to drone frame
            Eigen::Matrix3f rotation_matrix_pointcloud_to_drone;
            rotation_matrix_pointcloud_to_drone << 0, 0, 1,
                                                    -1, 0, 0,
                                                    0, -1, 0;

            // Filter out points below ground level after transformation
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_filtered->begin(); it != cloud_filtered->end(); ++it) {
                // Original coordinates in the point cloud frame
                Eigen::Vector3f point(it->x, it->y, it->z);

                // Transform from point cloud frame to drone frame
                Eigen::Vector3f point_in_drone_frame = rotation_matrix_pointcloud_to_drone * point;

                // Apply rotation based on quadrotor orientation and translation to global frame
                Eigen::Vector3f transformed_point;
                transformed_point(0) = rotation_matrix[0][0] * point_in_drone_frame(0) + rotation_matrix[0][1] * point_in_drone_frame(1) + rotation_matrix[0][2] * point_in_drone_frame(2) + tx - sensor_offset.pose.position.x;
                transformed_point(1) = rotation_matrix[1][0] * point_in_drone_frame(0) + rotation_matrix[1][1] * point_in_drone_frame(1) + rotation_matrix[1][2] * point_in_drone_frame(2) + ty - sensor_offset.pose.position.y;
                transformed_point(2) = rotation_matrix[2][0] * point_in_drone_frame(0) + rotation_matrix[2][1] * point_in_drone_frame(1) + rotation_matrix[2][2] * point_in_drone_frame(2) + tz - sensor_offset.pose.position.z;


                // Reject ground points in the transformed point cloud (represented in global frame)
                double ground_threshold = 0.3;  // Adjust this value to match the height of the ground
                if (transformed_point(2) > ground_threshold) {
                    pcl::PointXYZ pcl_point;
                    pcl_point.x = transformed_point(0);
                    pcl_point.y = transformed_point(1);
                    pcl_point.z = transformed_point(2);

                    transformed_cloud->push_back(pcl_point);

                    // Insert transformed point into octree
                    octree.updateNode(octomap::point3d(transformed_point(0), transformed_point(1), transformed_point(2)), true);
                }
            }

            // Convert filtered PCL PointCloud back to PCL PointCloud 
            *current_point_cloud = *transformed_cloud;

            // Prune the octree to optimize memory usage
            octree.updateInnerOccupancy();

            publishOctomap();

            data_received = true;
        }

        // Function to get the current point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr getPointClouds() {
            std::lock_guard<std::mutex> lock(mutex_);
            return current_point_cloud;
        }

        const octomap::OcTree& getOctoMap() const {
            return octree;
        }

        void publishOctomap() {
            // Convert the octree to a ROS Octomap message and publish
            octomap_msgs::Octomap octomap_msg;
            octomap_msg.header.frame_id = "map";
            octomap_msg.header.stamp = ros::Time::now();
            if (octomap_msgs::fullMapToMsg(octree, octomap_msg)) {
                octomap_pub.publish(octomap_msg);
            }
            else {
                ROS_ERROR("Failed to serialize the OctoMap.");
            }
        }

        bool isDataAvailable() {
            std::lock_guard<std::mutex> lock(mutex_);
            return data_received;
        }

        void clearDataAvailable() {
            data_received = false;
        }

    private:
        ros::NodeHandle nh;
        ros::Subscriber point_cloud_sub;
        ros::Publisher point_cloud_pub;
        ros::Publisher octomap_pub;

        pcl::PointCloud<pcl::PointXYZ>::Ptr current_point_cloud;
        // sensor_msgs::PointCloud2 current_point_cloud;
        geometry_msgs::PoseStamped sensor_offset;
        octomap::OcTree octree;
        std::mutex mutex_;
        QuadStatesScriber& quad_states_subscriber;

        bool data_received;
};

class ControlPublisher {
    public:
        ControlPublisher(ros::NodeHandle nh)
            : nh(nh) 
        {
            cmd_pub = nh.advertise<mavros_msgs::AttitudeTarget>("control_cmd", 10);
            ROS_INFO("[MPPI] Control inputs publisher has been successfully created!");
        }

        void controlPublish(mavros_msgs::AttitudeTarget& command) {
            cmd_pub.publish(command);
        }

    private:
        ros::NodeHandle nh;
        ros::Publisher cmd_pub;
};


int main(int argc, char** argv) {
    ros::init(argc, argv, "quad_mppi_node");
    ros::NodeHandle nh;

    geometry_msgs::PoseStamped start_point;
    geometry_msgs::PoseStamped goal_point;

    goal_point.pose.position.x = 0.0;
    goal_point.pose.position.y = 0.0;
    goal_point.pose.position.z = 2.0;

    QuadStatesScriber quad_states_subscriber(nh);

    // Wait for quadrotor state data to be available
    ros::Rate rate(10);  // 100 Hz
    while (ros::ok() && !quad_states_subscriber.isDataAvailable()) {
        ros::spinOnce();  // Process incoming messages
        rate.sleep();
    }
    ROS_INFO("[MPPI] Quadrotor data is available!");

    SensorDataManager depth_camera_manager(nh, quad_states_subscriber);
    MPPIController mppi_controller(nh, start_point, goal_point);
    ControlPublisher control_publisher(nh);

    pcl::PointCloud<pcl::PointXYZ>::Ptr current_point_cloud = depth_camera_manager.getPointClouds();
    while (ros::ok() && !depth_camera_manager.isDataAvailable()) {
        ros::spinOnce();
        rate.sleep();
    }
    depth_camera_manager.clearDataAvailable();
    ROS_INFO("[MPPI] Point cloud data is available!");

    // Initial state and control sequence
    nav_msgs::Odometry quad_current_states = quad_states_subscriber.getQuadStates();
    sensor_msgs::Imu quad_current_imu = quad_states_subscriber.getQuadImu();
    quad_current_states = quad_states_subscriber.getQuadStates();
    start_point.pose.position.x = quad_current_states.pose.pose.position.x;
    start_point.pose.position.y = quad_current_states.pose.pose.position.y;
    start_point.pose.position.z = quad_current_states.pose.pose.position.z;

    std::vector<Controls> control_sequence(MPPI::N, {QuadParams::mass*QuadParams::gravConst, 0.0, 0.0, 0.0});

    while (ros::ok()) {

        // current states receive
        quad_current_states = quad_states_subscriber.getQuadStates();
        quad_current_imu = quad_states_subscriber.getQuadImu();
        quad_current_states.twist.twist.angular.x = quad_current_imu.angular_velocity.x;
        quad_current_states.twist.twist.angular.y = quad_current_imu.angular_velocity.y;
        quad_current_states.twist.twist.angular.z = quad_current_imu.angular_velocity.z;

        // Get point clouds for obstacles
        if (depth_camera_manager.isDataAvailable()) {
            depth_camera_manager.clearDataAvailable();
        }

        // Perform MPPI optimization
        mppi_controller.mppi_optimization(quad_current_states, control_sequence, current_point_cloud);
        std::vector<nav_msgs::Odometry>& optimal_states = mppi_controller.get_optimal_states();

        // Publish the first control in the sequence
        mavros_msgs::AttitudeTarget cmd;

        // Update state based on noisy control
        cmd.orientation.w = optimal_states[1].pose.pose.orientation.w;
        cmd.orientation.x = optimal_states[1].pose.pose.orientation.x;
        cmd.orientation.y = optimal_states[1].pose.pose.orientation.y;   
        cmd.orientation.z = optimal_states[1].pose.pose.orientation.z;
        cmd.thrust = std::min(control_sequence[0].ct / MAX_THRUST, 1.0);
        control_publisher.controlPublish(cmd);
        ROS_INFO("Controls: thrust:%f, quaternions: %f, %f, %f, %f", cmd.thrust, cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z);

        // Shift the control sequence and reinitialize the last one
        for (int i = 0; i < MPPI::N - 1; ++i) {
            control_sequence[i] = control_sequence[i + 1];
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}