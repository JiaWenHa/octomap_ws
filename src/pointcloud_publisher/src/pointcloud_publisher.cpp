/**
*
* 读取pcd点云文件并发布到topic
*
*/
#include<iostream>
#include<string>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <vector>

#include<ros/ros.h>  
#include<pcl/point_cloud.h>  
#include<pcl_conversions/pcl_conversions.h>  
#include<sensor_msgs/PointCloud2.h>  
#include<pcl/io/pcd_io.h>
#include<pcl/filters/radius_outlier_removal.h>

using namespace std;

int main (int argc, char **argv)  
{  
	std::string topic,path,frame_id;
    int hz=5;

	ros::init (argc, argv, "pointcloud_publisher");  
	ros::NodeHandle nh("~");  

    nh.param<std::string>("path", path, "/home/ros/Programs/octomap_ws/src/pointcloud_publisher/data/map.pcd");
	nh.param<std::string>("frame_id", frame_id, "map");
	nh.param<std::string>("topic", topic, "/pointcloud/output");
    nh.param<int>("hz", hz, 5);
   
	ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2> (topic, 10);  

	pcl::PointCloud<pcl::PointXYZ> cloud;  
	sensor_msgs::PointCloud2 output;  
	pcl::io::loadPCDFile (path, cloud);  

	/**
	 * 增加半径滤波算法
	 */ 
	// 输入待滤波的原始点云指针
	pcl::PointCloud<pcl::PointXYZ>::Ptr true_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	// 保存滤波后的点云指针
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	// 创建滤波器对象
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	// 设置要滤波的点云
	*true_cloud = cloud;
	outrem.setInputCloud(true_cloud);
	// 设置滤波半径
	outrem.setRadiusSearch(0.1);
	// 设置滤波最小近邻数
	outrem.setMinNeighborsInRadius(10);
	// 执行半径滤波
	outrem.filter(*cloud_filtered);


	pcl::toROSMsg(*cloud_filtered,output);// 转换成ROS的数据类型, 通过topic发布

	output.header.stamp=ros::Time::now();
	output.header.frame_id  =frame_id;

	cout<<"path = "<<path<<endl;
	cout<<"frame_id = "<<frame_id<<endl;
	cout<<"topic = "<<topic<<endl;
	cout<<"hz = "<<hz<<endl;

	ros::Rate loop_rate(hz);  
	while (ros::ok())  
	{  
		pcl_pub.publish(output);  
		ros::spinOnce();  
		loop_rate.sleep();  
	}  
	return 0;  
}  
