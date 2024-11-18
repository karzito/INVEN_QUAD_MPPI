/**
 * @file offb_node.cpp
 * @brief Offboard control example node, written with MAVROS version 0.19.x, PX4 Pro Flight
 * Stack and tested in Gazebo Classic SITL
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/RCOut.h>


mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

mavros_msgs::RCOut current_pwm;
void rcOut_cb(const mavros_msgs::RCOut::ConstPtr& msg) {
    current_pwm = *msg;
}

geometry_msgs::Twist current_cmd_vel;
bool cmd_vel_received = false;
void cmd_vel_cb(const geometry_msgs::Twist::ConstPtr& msg) {
    current_cmd_vel = *msg;
    cmd_vel_received = true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "my_quad_node");
    ros::NodeHandle nh;

    // general variables
    bool tracking_flag = false;

    // ros topics and services
	ros::Subscriber rcOut_sub = nh.subscribe<mavros_msgs::RCOut>
			("mavros/rc/out", 10, rcOut_cb);
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    ros::Publisher attitude_target_pub = nh.advertise<mavros_msgs::AttitudeTarget>
            ("mavros/setpoint_raw/attitude", 10);
    ros::Subscriber cmd_vel_sub = nh.subscribe<geometry_msgs::Twist>
            ("/cmd_vel", 10, cmd_vel_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("mavros/setpoint_position/local", 10);
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::TwistStamped>
            ("mavros/setpoint_velocity/cmd_vel", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(100.0);

    // wait for FCU connection
    while(ros::ok() && !current_state.connected) {
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    geometry_msgs::PoseStamped pos_cmd_stamped;
    pos_cmd_stamped.pose.position.x = 0.0;
    pos_cmd_stamped.pose.position.y = 0.0;
    pos_cmd_stamped.pose.position.z = 2.0;

    geometry_msgs::TwistStamped vel_cmd_stamped;
    vel_cmd_stamped.twist.linear.x = 0;
    vel_cmd_stamped.twist.linear.y = 0;
    vel_cmd_stamped.twist.linear.z = 0;

    ros::Time last_request = ros::Time::now();

    while(ros::ok()) {
        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(2.0))) {
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent) {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(2.0))) {
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success) {
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }

        if ( current_state.armed && cmd_vel_received && (ros::Time::now() - last_request > ros::Duration(10.0)) ) {
            if (!tracking_flag) {
                ROS_INFO("MPPI Control Start!");
                tracking_flag = true;
            }
            else {
                // Update velocity command based on received command from quad_mppi_node
                vel_cmd_stamped.header.stamp = ros::Time::now();
                vel_cmd_stamped.twist = current_cmd_vel;
            }
        }

        if (tracking_flag) vel_pub.publish(vel_cmd_stamped);
        else pos_pub.publish(pos_cmd_stamped);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}