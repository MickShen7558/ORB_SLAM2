/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include<Frame.h>
#include<ORBmatcher.h>
#include<Optimizer.h>

using namespace std;

void LoadImages(string &strPath, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps, bool beKitti);
std::string getDirEnd(std::string dataset_dir);
ORB_SLAM2::Frame* getFrame(const string &strImageLeft,
                           const string &strImageRight,
                           ORB_SLAM2::ORBVocabulary *mpVocabulary,
                           const cv::Mat& M1l,
                           const cv::Mat& M2l,
                           const cv::Mat& M1r,
                           const cv::Mat& M2r,
                           const cv::FileStorage& fsSettings,
                           bool beKitti);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings dataset_path dataset_source" << endl;
        return 1;
    }

    //TODO: 1. Read in two pairs of stereo images from a directory
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    string dataset_type = string(argv[4]);
    string dataset_path = string(argv[3]);
    LoadImages(dataset_path, vstrImageLeft, vstrImageRight, vTimeStamp, dataset_type == "kitti");
    if(vstrImageLeft.size() < 2 || vstrImageRight.size() < 2)
    {
        cerr << "ERROR: No enough images in provided path." << endl;
        return 1;
    }
    vstrImageLeft.resize(2);
    vstrImageRight.resize(2);



    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    auto* mpVocabulary = new ORB_SLAM2::ORBVocabulary();
    //获取字典加载状态
    bool bVocLoad = mpVocabulary->loadFromTextFile(argv[1]);
    //如果加载失败，就输出调试信息
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << argv[1] << endl;
        //然后退出
        exit(-1);
    }

    //Create KeyFrame Database
    auto mpKeyFrameDatabase = new ORB_SLAM2::KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    auto mpMap = new ORB_SLAM2::Map();

    cv::Mat M1l,M2l,M1r,M2r;
    if(dataset_type != "kitti") {
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r,T_lr,R_lr;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;


        fsSettings["TranslationOfCamera2"] >> T_lr;
        fsSettings["RotationOfCamera2"] >> R_lr;
        //cout<<R_lr<<endl;
        //R_lr=R_lr.t();
        //cout<<R_lr<<endl;
        //T_lr = T_lr/10;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
           rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::Size imageSize(cols_l,rows_l);
        //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
        cv::Mat Rl, Rr, Pl, Pr, Q;
        //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域, 其内部的所有像素都有效
        cv::Rect validROIL;
        cv::Rect validROIR;
        //经过双目标定得到摄像头的各项参数后，采用OpenCV中的stereoRectify(立体校正)得到校正旋转矩阵R、投影矩阵P、重投影矩阵Q
        //flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
        //alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
        stereoRectify(K_l, D_l, K_r, D_r, imageSize, R_lr, T_lr, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY,
                      0, imageSize, &validROIL, &validROIR);
        //cout<<Pl<<endl;
        //cout<<Pr<<endl;
        // 相机校正
        //再采用映射变换计算函数initUndistortRectifyMap得出校准映射参数,该函数功能是计算畸变矫正和立体校正的映射变换
        cv::initUndistortRectifyMap(K_l,D_l,Rl,Pl.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        cv::initUndistortRectifyMap(K_r,D_r,Rr,Pr.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
        //cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        //cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
        //cv::initUndistortRectifyMap(K_l,D_l,R_l,cv::Mat(),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        //cv::initUndistortRectifyMap(K_r,D_r,R_r,cv::Mat(),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    }




    //TODO: 2. Extract the features of the pair of images of the first pair

    ORB_SLAM2::Frame* first = getFrame(vstrImageLeft[0], vstrImageRight[0], mpVocabulary, M1l, M2l, M1r, M2r, fsSettings, dataset_type == "kitti");
    // Set Frame pose to the origin
    cout << "The number of KeyPoints in the first frame is " << first->N << endl;

    first->SetPose(cv::Mat::eye(4,4,CV_32F));

    // Create KeyFrame
    // step 2：将当前帧构造为初始关键帧
    // mCurrentFrame的数据类型为Frame
    // KeyFrame包含Frame、地图3D点、以及BoW
    // KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap
    // KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpMap都指向Tracking里的这个mpKeyFrameDB
    // 提问: 为什么要指向Tracking中的相应的变量呢? -- 因为Tracking是主线程，是它创建和加载的这些模块
    auto* pKFini = new ORB_SLAM2::KeyFrame(*first, mpMap, mpKeyFrameDatabase);

    // Create MapPoints and asscoiate to KeyFrame
    // step 4：为每个特征点构造MapPoint
    for(int i=0; i<first->N;i++)
    {
        //只有具有正深度的点才会被构造地图点
        float z = first->mvDepth[i];
        if(z>0)
        {
            // step 4.1：通过反投影得到该特征点的3D坐标
            cv::Mat x3D = first->UnprojectStereo(i);
            // step 4.2：将3D点构造为MapPoint
            auto* pNewMP = new ORB_SLAM2::MapPoint(x3D,pKFini,mpMap);

            // step 4.3：为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            // b.该MapPoint的描述子
            // c.该MapPoint的平均观测方向和深度范围

            // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
            pNewMP->AddObservation(pKFini,i);  //双目、rgbd添加的是两个
            // b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
            //? 如何定义的这个区分度?
            pNewMP->ComputeDistinctiveDescriptors();
            // c.更新该MapPoint平均观测方向以及观测距离的范围
            pNewMP->UpdateNormalAndDepth();

            // step 4.4：在地图中添加该MapPoint
            // mpMap->GetAllMapPoints()
            mpMap->AddMapPoint(pNewMP);
            // step 4.5：表示该KeyFrame的哪个特征点可以观测到哪个3D点
            pKFini->AddMapPoint(pNewMP,i);

            // step 4.6：将该MapPoint添加到当前帧的mvpMapPoints中
            // 为当前Frame的特征点与MapPoint之间建立索引
            first->mvpMapPoints[i]=pNewMP;
        }
    }

    cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;


    //TODO: 3. Extract the features of the pair of images of the second frame
    //

    ORB_SLAM2::Frame* second = getFrame(vstrImageLeft[1], vstrImageRight[1], mpVocabulary, M1l, M2l, M1r, M2r, fsSettings, dataset_type == "kitti");


    //TODO: 4. Do feature mapping to the first frame and get the relative position.
    ORB_SLAM2::ORBmatcher matcher(0.9, true);

    //Tracking::UpdateLastFrame()

    // step 2：对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
    // 注意这些MapPoints不加入到Map中，在tracking的最后会删除
    // 跟踪过程中需要将将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // step 2.1：得到上一帧有深度值的特征点
    //第一个元素是某个点的深度,第二个元素是对应的特征点id
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(first->N);

    for(int i=0; i<first->N;i++)
    {
        float z = first->mvDepth[i];
        if(z>0)

        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    //如果上一帧中没有有效深度的点,那么就直接退出了
    if(vDepthIdx.empty())
        exit(-2);

    // step 2.2：按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // step 2.3：将距离比较近的点包装成MapPoints
    int nPoints = 0;

    float fx = fsSettings["Camera.fx"];

    float bf = fsSettings["Camera.bf"];

    float ThDepth = bf*(float)fsSettings["ThDepth"]/fx;

    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        //如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
        ORB_SLAM2::MapPoint* pMP = first->mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)      //? 从地图点被创建后就没有观测到,意味这是在上一帧中新添加的地图点吗
        {
            bCreateNew = true;
        }

        //如果需要创建新的临时地图点
        if(bCreateNew)
        {
            // 这些生UpdateLastFrameints后并没有通过：
            // a.AddMaUpdateLastFrame、
            // b.AddObUpdateLastFrameion、
            // c.CompuUpdateLastFrameinctiveDescriptors、
            // d.UpdatUpdateLastFramelAndDepth添加属性，
            // 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率   -- 我觉得可以这么说是因为在临时地图中增加了地图点，能够和局部地图一并进行定位工作
            cv::Mat x3D = first->UnprojectStereo(i);
            auto* pNewMP = new ORB_SLAM2::MapPoint(
                    x3D,            //该点对应的空间点坐标
                    mpMap,          //? 不明白为什么还要有这个参数
                    first,          //存在这个特征点的帧(上一帧)
                    i);             //特征点id

            //? 上一帧在处理结束的时候,没有进行添加的操作吗?
            first->mvpMapPoints[i]=pNewMP; // 添加新的MapPoint

            nPoints++;
        }
        else//如果不需要创建新的 临时地图点
        {
            nPoints++;
        }


        //当当前的点的深度已经超过了远点的阈值,并且已经这样处理了超过100个点的时候,说明就足够了
        if(vDepthIdx[j].first>ThDepth && nPoints>100)
            break;
    }
    //end Tracking::UpdateLastFrame()

    // set the initial position the same as the first frame since we made the assumption that the relative motion is small.
    second->SetPose(cv::Mat::eye(4,4,CV_32F));
    //清空当前帧的地图点
    fill(second->mvpMapPoints.begin(),second->mvpMapPoints.end(),static_cast<ORB_SLAM2::MapPoint*>(nullptr));

    int th = 7;

    // step 2：根据匀速度模型进行对上一帧的MapPoints进行跟踪, 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    //我觉的这个才是使用恒速模型的根本目的
    int nmatches = matcher.SearchByProjection(*second,*first,(float)th,false);

    // If few matches, uses a wider window search
    // 如果跟踪的点少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(second->mvpMapPoints.begin(),second->mvpMapPoints.end(),static_cast<ORB_SLAM2::MapPoint*>(nullptr));
        nmatches = matcher.SearchByProjection(*second,*first,(float)th*2,false); // 2*th
    }

    //如果就算是这样还是不能够获得足够的跟踪点,那么就认为运动跟踪失败了.
    if(nmatches<20)
        cout << "Too few matched points, the number of match is " << nmatches << endl;

    // Optimize frame pose with all matches
    // step 3：优化位姿
    ORB_SLAM2::Optimizer::PoseOptimization(second);

    cout << "Tcl = " << endl << " "  << second->mTcw << endl << endl;

    return 0;
}

#include <sys/stat.h>
inline bool exists_file (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

std::string getDirEnd(std::string dataset_dir)
{
    //useless in this part
    std::string end;
    unsigned int iSize = dataset_dir.size();
    unsigned int i = 0;
    for(i = 0; i < iSize; i++)
    {
        if(dataset_dir.at(i)=='/' && i!=iSize-1)
            end=dataset_dir.substr(i+1);
    }
    if (end[end.size()-1]=='/')
        end.pop_back();
    return end;
}


void LoadImages(string &strPath, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps, bool beKitti)
{
    cerr << "Start LoadImages." << endl;
    //vTimeStamps.reserve(10000);
    //vstrImageLeft.reserve(10000);
    //vstrImageRight.reserve(10000);

    unsigned int iSize = strPath.size();
    if(strPath.at(iSize-1)!='/')
        strPath.push_back('/');

    string strPathLeft = strPath + "left";
    string strPathRight = strPath + "right";
    if(beKitti) {
        int i = 0;
        do{
            stringstream ss;
            ss << setfill('0') << setw(6) << i;
            std::string file = strPathLeft + "/" + ss.str() + ".png";
            //if (i == 0)
            //    cout << file << endl;
            if(exists_file(file))
            {
                double t = i;
                vTimeStamps.push_back(t);
                ss.clear();ss.str("");
                ss << setfill('0') << setw(6) << i;
//                if (i == 0)
//                    cout << strPathLeft + "/" + ss.str() + ".png\n";
                vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
                ss.clear();ss.str("");
                ss << setfill('0') << setw(6) << i;
                vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
                i++;
            }
            else
                break;
        }while(1);
    }
    else {
        int left_i=0, right_i=1;
        do{
            stringstream ss;
            ss << setfill('0') << setw(6) << left_i;
            std::string file = strPathLeft + "/" + ss.str() + ".jpg";
            if(exists_file(file))
            {
                double t = 0.05*left_i;
                vTimeStamps.push_back(t);
                ss.clear();ss.str("");
                ss << setfill('0') << setw(6) << left_i;
                vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".jpg");
                ss.clear();ss.str("");
                ss << setfill('0') << setw(6) << right_i;
                vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".jpg");
                left_i = left_i + 2;
                right_i = right_i + 2;
            }
            else
                break;
        }while(1);
    }

    cout<<"Finish LoadImages: "<<vstrImageLeft.size()<<endl;
}

ORB_SLAM2::Frame* getFrame(const string &strImageLeft,
                          const string &strImageRight,
                          ORB_SLAM2::ORBVocabulary *mpVocabulary,
                          const cv::Mat& M1l,
                          const cv::Mat& M2l,
                          const cv::Mat& M1r,
                          const cv::Mat& M2r,
                          const cv::FileStorage& fsSettings,
                          bool beKitti){
    cv::Mat imLeft, imRight, imLeftRect, imRightRect, imGrayLeft, imGrayRight;

    imLeft = cv::imread(strImageLeft,CV_LOAD_IMAGE_UNCHANGED);
    imRight = cv::imread(strImageRight,CV_LOAD_IMAGE_UNCHANGED);
    if(imLeft.empty())
    {
        cerr << endl << "Failed to load image at: "
             << string(strImageLeft) << endl;
        exit(1);
    }
    if(imRight.empty())
    {
        cerr << endl << "Failed to load image at: "
             << string(strImageRight) << endl;
        exit(1);
    }

    if(!beKitti){
        //remap for none kitti
        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
        //Change to gray images, only for none kitti
        cvtColor(imLeftRect,imGrayLeft,CV_RGBA2GRAY);
        cvtColor(imRightRect,imGrayRight,CV_RGBA2GRAY);
    }
    else{
        imLeft.copyTo(imLeftRect);
        imRight.copyTo(imRightRect);
    }

    if (imLeftRect.channels() == 3) {
        if ((int)fsSettings["Camera_RGB"]) {
            cvtColor(imLeftRect,imGrayLeft,CV_RGBA2GRAY);
            cvtColor(imRightRect,imGrayRight,CV_RGBA2GRAY);
        } else {
            cvtColor(imLeftRect,imGrayLeft,CV_BGRA2GRAY);
            cvtColor(imRightRect,imGrayRight,CV_BGRA2GRAY);
        }
    } else if (imLeftRect.channels() == 4) {
        if ((int)fsSettings["Camera_RGB"]) {
            cvtColor(imLeftRect,imGrayLeft,CV_RGBA2GRAY);
            cvtColor(imRightRect,imGrayRight,CV_RGBA2GRAY);
        } else {
            cvtColor(imLeftRect,imGrayLeft,CV_BGRA2GRAY);
            cvtColor(imRightRect,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // step 2 加载ORB特征点有关的参数,并新建特征点提取器

    // 每一帧提取的特征点数 1000
    int nFeatures = fsSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fsSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fsSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fsSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fsSettings["ORBextractor.minThFAST"];

    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    auto* ORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    auto* ORBextractorRight = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);



    float fx = fsSettings["Camera.fx"];
    float fy = fsSettings["Camera.fy"];
    float cx = fsSettings["Camera.cx"];
    float cy = fsSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    //构造相机内参矩阵
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fsSettings["Camera.k1"];
    DistCoef.at<float>(1) = fsSettings["Camera.k2"];
    DistCoef.at<float>(2) = fsSettings["Camera.p1"];
    DistCoef.at<float>(3) = fsSettings["Camera.p2"];
    const float k3 = fsSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    float bf = fsSettings["Camera.bf"];

    float ThDepth = bf*(float)fsSettings["ThDepth"]/fx;

    auto* myFrame = new ORB_SLAM2::Frame(
            imGrayLeft,             //左目图像
            imGrayRight,            //右目图像
            0,                      //时间戳
            ORBextractorLeft,       //左目特征提取器
            ORBextractorRight,      //右目特征提取器
            mpVocabulary   ,        //字典
            K,                      //内参矩阵
            DistCoef,               //去畸变参数
            bf,                     //基线长度
            ThDepth               //远点,近点的区分阈值
    );

    return myFrame;
}

