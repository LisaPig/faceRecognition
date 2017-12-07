#include<opencv2/opencv.hpp>  
#include<iostream>  
#include<vector>  
#include<string>  
#include<windows.h> 
#include<stdio.h>  //sprintf_s
#include <KL1p.h> 
#include"armadillo"

using namespace std;  
using namespace cv;  
using namespace kl1p; 
 

bool traverseORL(vector<string> &filenames,vector<int> &labels,bool bTrain);

/***************************************   main()  *******************************************/
int main()  
{  
   vector<Mat> faces;
   vector<string> filenames;
   vector<int> labels;
   bool bTrain=true; //false;//true;

/***************************************   traverseORL *******************************************/

   int64 timeCount1 = getTickCount();       //int64  typedef long long int64 
   traverseORL(filenames,labels, bTrain);  //read ORL files （eg. s1\1.pgm->s40\1.pgm）
   timeCount1 = static_cast<int64>((getTickCount() - timeCount1)*1000.0  / getTickFrequency());   //getTickFrequency() 2.53331e006---2.5M
   cout << "traverseORL files spend time: " << timeCount1 << " ms" << endl;  //700ms
   cout<<getTickFrequency()<<endl;

/***************************************   read files to faces(Mat) *******************************************/
   int64 timeCount2 = getTickCount();    
   for(int i=0;i<filenames.size();i++) // read files to faces(Mat)
	   faces.push_back( imread(filenames[i]) );

    timeCount2 = static_cast<int64>((getTickCount() - timeCount2) *1000.0 / getTickFrequency() );  
    cout << "imread 200 files spend time: " << timeCount2 << " ms" << endl;   //60ms
	cout<<getTickFrequency()<<endl;

/***************************************   read files to faces(Mat) *******************************************/
	imshow("face1",faces[199]);


   while(waitKey(10)!=27) ;

return true;
}  

/*
function：read ORL files （eg. s1\1.pgm）
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
  for(int i=0;i<filenames.size();i++)
	  cout<<filenames[i]<<endl;

 return true;	
 

}


