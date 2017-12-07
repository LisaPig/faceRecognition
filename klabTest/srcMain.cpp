//SRC

#include<opencv2/opencv.hpp>  
#include<iostream>  
#include<vector>  
#include<string>  
#include<windows.h> 
#include<stdio.h>  //sprintf_s
#include <KL1p.h> 
#include"armadillo"
#include <KL1pInclude.h>

using namespace std;  
using namespace cv;  
using namespace kl1p; 
 

bool traverseORL(vector<string> &filenames,vector<int> &labels,bool bTrain);
void	WriteToCSVFile(const arma::Col<klab::DoubleReal>& signal, const std::string& filePath);
// ---------------------------------------------------------------------------------------------------- //

/***************************************   main()  *******************************************/
int main()  
{  
   vector<Mat> images;
   vector<string> filenames;
   vector<int> labels;
   bool bTrain=true; //false;//true;
  // Mat testImage=imread("F:\\Visual2012\\klabTest\\klabTest\\ORL\\s40\\7.pgm");
   Mat testImage=imread("testSet\\salt_pepper_Image.pgm");
/***************************************   traverseORL *******************************************/
traverseORL(filenames,labels, bTrain);  //read ORL files £¨eg. s1\1.pgm->s40\1.pgm£©

/***************************************   read files to faces(Mat) *******************************************/
 for(size_t i=0;i<filenames.size();i++) // read files to faces(Mat)   //size_t == unsigned int
	   images.push_back( imread(filenames[i]) );
 
 cout<<"read files to faces is OK!"<<endl;
/***************************************   read files to faces(Mat) *******************************************/
	//imshow("face1",faces[199]);
 //initial arma::Mat  
	arma::Mat<klab::DoubleReal> A(10304,200);
	arma::Mat<klab::DoubleReal> A1(500,200);
	arma::Mat<klab::DoubleReal> R(500,10304);
	arma::Col<klab::DoubleReal> Y(10304);
	arma::Col<klab::DoubleReal> Y1(500);
	arma::Mat<klab::DoubleReal> B(500,700);
	arma::Col<klab::DoubleReal> W(700);
	arma::Col<klab::DoubleReal> x1(200);
	arma::Col<klab::DoubleReal> e1(500);
	Mat image,image1;
	double sum;
	int rowA=0;
/***********************************	Create random gaussian i.i.d matrix R of size (m,n). *****************/
	/*int m=500,n=10304;
	klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal> > R =
		new kl1p::TNormalRandomMatrixOperator<klab::DoubleReal>(m, n, 0.0, 1.0);
   */
	int64 timeCount1 = getTickCount();       //int64  typedef long long int64 
	//read R.txt (gaussian random matrix)
	fstream f("R.txt",ios::in);
	for(int i=0;i<500;i++)
		for(int j=0;j<10304;j++)
			f>>R(i,j);
	f.close();
   timeCount1 = static_cast<int64>((getTickCount() - timeCount1)*1000.0  / getTickFrequency());   //getTickFrequency() 2.53331e006---2.5M
   cout << "traverseORL files spend time: " << timeCount1 << " ms" << endl;  // 160594ms
	
/**********************************************************  construct Matrix A  *****************************/
	for(int k=0;k<200;k++) // trainImage 200
	{
	rowA=0;
	image=images[k];
	if(image.channels()==3)//multi-channel
	{
		cvtColor(image,image1,CV_BGR2GRAY);
		if(image1.channels()==1)
			image1.copyTo(image);
	}
	// normalization by 2-norm
	sum=0;
	for(int i=0;i<image.cols;i++)
		for(int j=0;j<image.rows;j++)
		{ sum+=image.at<uchar>(j,i)*image.at<uchar>(j,i);}
	
	sum=sqrt(sum);  // 2-norm
	for(int i=0;i<image.cols;i++)
		for(int j=0;j<image.rows;j++)
		{ A(rowA++,k)=image.at<uchar>(j,i)/sum;} //normalization
	} // A :10304*200

	cout<<"construct Matrix A is OK!"<<endl;
/************************************************ construct Matrix A1   *****************************/
	A1=R*A; // 500*200

	  for(int i=0;i<200;i++)   // 2-norm based on col
    {  
        sum=0;  
        for(int j=0;j<500;j++)  
        {  
            sum=sum+A1(j,i)*A1(j,i);  
        }  
        sum=sqrt(sum);  
        for(int j=0;j<500;j++)  
        {  
            A1(j,i)=A1(j,i)/sum;  
        }  
    }  // A1:500*200
  cout<<"construct Matrix A1 is OK!"<<endl;
/************************************************* construct Matrix B **********************************************/
	for(int i=0;i<500;i++)
		for(int j=0;j<700;j++)
		{
			if(j<200) B(i,j)=A1(i,j);
			else   
			{ if(j==i+200) B(i,j)=1;
			  else         B(i,j)=0;
			}
		} // B:500*700

	cout<<"construct Matrix B is OK!"<<endl;
/*********************************************** construct testImage Matrix Y  **********************************************/
   int rowY=0;
   imshow("testImage",testImage);
   image=testImage;
	if(image.channels()==3)//multi-channel
	{
		cvtColor(image,image1,CV_BGR2GRAY);
		if(image1.channels()==1)
			image1.copyTo(image);
	}
	// normalization by 2-norm
	sum=0;
	for(int i=0;i<image.cols;i++)
		for(int j=0;j<image.rows;j++)
		{sum+=image.at<uchar>(j,i)*image.at<uchar>(j,i);}
	
	sum=sqrt(sum);  // 2-norm

	for(int i=0;i<image.cols;i++)
		for(int j=0;j<image.rows;j++)
		{ Y(rowY++)=image.at<uchar>(j,i)/sum;} //normalization     Y(rowY++,0) one col

		cout<<"construct Matrix Y  is OK!"<<endl;
/********************************************* construct  Matrix Y1   **********************************************/
		Y1=R*Y;
		
		sum=0;  
        for(int j=0;j<500;j++)  
        {  
            sum=sum+Y1(j)*Y1(j);  
        }  
        sum=sqrt(sum);  
        for(int j=0;j<500;j++)  
        {  
            Y1(j)=Y1(j)/sum;  
        }  
cout<<"construct Matrix Y1 is OK!"<<endl;
/**********************************************  compute matrix W , solution of Y=BW **********************************************/
    kl1p::TMatrixOperator<klab::DoubleReal> * matrix = new kl1p::TMatrixOperator<klab::DoubleReal>(B);  
    klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal, klab::DoubleReal> > * B1 =
			new klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal, klab::DoubleReal> >(matrix);  

	int64 timeCount2 = getTickCount();       //int64  typedef long long int64
	double tolerance=1e-3; //tolerance of the solution
	
	/*  BasisPursuit
	kl1p::TBasisPursuitSolver<klab::DoubleReal> bp(tolerance);  // BasisPursuit:55957ms  (sci=0.574479,tolerance=1e-3)
	bp.solve(Y1,*B1,W);  // *B1-> SmartPointer
	//WriteToCSVFile(W, "BasisPursuit-Signal.csv");	// Write solution to a file.
	*/
	
	// OMP......
	klab::DoubleReal rho = 0.1;				// Ratio of the sparsity of the signal x0.
	int  row_n=10304;
	klab::UInt32 kk = klab::UInt32(rho*row_n);	// Sparsity of the signal x0 (number of non-zero elements).
	

	//   OMP
	//klab::DoubleReal rho = 0.1;				// Ratio of the sparsity of the signal x0.
	//int  row_n=10304;
	//klab::UInt32 kk = klab::UInt32(rho*row_n);	// Sparsity of the signal x0 (number of non-zero elements).
	kl1p::TOMPSolver<klab::DoubleReal> omp(tolerance);
	omp.solve(Y1,*B1,kk,W);   // OMP:32418ms (  kk=10304*0.1;tolerance=1e-3;sci=1)
	//*/

	/*  ROMP
	kl1p::TROMPSolver<klab::DoubleReal> romp(tolerance);
	romp.solve(Y1,*B1,kk,W);  // ROMP:5007ms,sci=0.0375417
	*/
	/*  CoSaMP
	kl1p::TCoSaMPSolver<klab::DoubleReal> cosamp(tolerance);
	cosamp.solve(Y1,*B1,kk,W); //  CoSaMP: ms
	WriteToCSVFile(W, "CoSaMP-Signal.csv");	// Write solution to a file.
	*/
   timeCount2 = static_cast<int64>((getTickCount() - timeCount2)*1000.0  / getTickFrequency());   //getTickFrequency() 2.53331e006---2.5M
   cout << "compute 1-norm spend time: " << timeCount2<< " ms" << endl;  // 358ms
   cout<<"compute matrix W is OK!"<<endl;
/*******************************************************   solution of  matrix Y    **********************************************/
	for(int i=0;i<700;i++)
	{
		if(i<200) x1(i)=W(i);  // coefficient Matrix
		else      e1(i-200)=W(i); // occlusion(error) Matrix
	}
	Y1=Y1-e1; //cleaning image

	WriteToCSVFile(x1, "x1_BasisPursuit-Signal.csv");
	WriteToCSVFile(e1, "e1_BasisPursuit-Signal.csv");
	cout<<"solution of  matrix Y is OK!"<<endl;
/*******************************************************   Seeking residuals  **********************************************/
	double r[40]; // residuals
	arma::Col<double> I(200);
	arma::Col<double> I1(500);
	for(int k=0;k<40;k++) // 40 classes
	{
		sum=0;
		for(int j=0;j<200;j++) // A total of 200 TrainImages
			I(j)=0;
		for(int i=k*5;i<k*5+5;i++)
			I(i)=x1(i);          // sparse representation
		I1=Y1-A1*I;
		// residuals: compute the 2-norm of I1
		sum=0;
		for(int i=0;i<500;i++)
			sum+=I1(i)*I1(i);
		sum=sqrt(sum);
		r[k]=sum;
	}
	
	cout<<"Seeking residuals  is OK!"<<endl;
	
/***********************************************************  Seeking min_residuals **********************************************/
	double minNum=r[0];
	int minIndex=0;
	for(int i=0;i<40;i++)
	{
		if(r[i]<minNum)
		{
			minNum=r[i];
			minIndex=i;
		}
	}
	cout<<"the class of testImage is:"<<minIndex+1<<endl;
/**********************************************************   compute SCI(x)  **********************************************/
	/*  1.sparsity concentration index (SCI)
	    2.The SCI of a coefficient vector x 
	*/
	double sum1,sum2;
	double sum1_max=0;
	int sum1_maxIndex=0;
	for(int k=0;k<40;k++)
	{
		sum1=0;
		for(int i=k*5;i<k*5+5;i++)
		   sum1+=abs(x1(i));   // compute 1-norm of delt(xi)
		if(sum1>sum1_max) 
		{  sum1_max=sum1;
		   sum1_maxIndex=k;
		}
	
	}
	sum2=0;  
	for(int i=0;i<200;i++)
		sum2+=abs(x1(i));     // compute 1-norm of x

	int k=40; // 40 classes
	double sci=(k*sum1_max/sum2-1)/(k-1);
	cout<<"maxIndex:"<<sum1_maxIndex+1<<endl;
	cout<<"sci:"<< sci<<endl;  // 0.00980309

	while(waitKey(10)!=27) ;

return true;
}  


/*
function£ºread ORL files £¨eg. s1\1.pgm£©
parameter:
		--vector<string> &filenames : filepath
		--vector<int> &labels: the labels of faces
		--bool bTrain :true represents Train (No.1-No.5);
					   false represents Test(No.6-No.10)
commentray: sprintf_s write the filepath to strFilename 
*/
bool traverseORL(vector<string> &filenames,vector<int> &labels,bool bTrain)
{
  char strFilename[100];
  int k=0;
  for(int i=1;i<=40;i++) //40 filefolders
{
  if (bTrain) // true represents Train (No1-No5)
  {
	for(int j=1;j<=5;j++){
     sprintf_s(strFilename, "F:\\Visual2012\\klabTest\\klabTest\\ORL\\s%d\\%d.pgm", i,j); //sprintf_s write the filepath to strFilename 
     filenames.push_back( strFilename ); // vector
	 labels.push_back( i );
	}
  }
  else // false represents Test(No.6-No.10)
  {
    for(int j=6;j<=10;j++){
     sprintf_s(strFilename, "F:\\Visual2012\\klabTest\\klabTest\\ORL\\s%d\\%d.pgm", i,j); //sprintf_s write the filepath to strFilename 
     filenames.push_back( strFilename );
	 labels.push_back( i );
	}
  }
 }
  /*
  for(int i=0;i<filenames.size();i++)
	  cout<<filenames[i]<<endl;
	  */
 return true;	
 
}


 
void	WriteToCSVFile(const arma::Col<klab::DoubleReal>& signal, const std::string& filePath)
{
	std::ofstream of(filePath.c_str());
	if(of.is_open())
	{
		for(klab::UInt32 i=0; i<signal.n_rows; ++i)
			of<<i<<";"<<signal[i]<<std::endl;

		of.close();
	}
	else
	{
		std::cout<<"ERROR! Unable to open file \""<<filePath<<"\" !"<<std::endl;
	}
}