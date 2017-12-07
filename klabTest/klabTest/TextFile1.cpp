//KL1p Demo
#include<iostream>
#include"armadillo"
#include <KL1pInclude.h>
#include <KL1p.h>

using namespace std;
using namespace kl1p;
using namespace arma;


#if 1
int main()
{
    arma::mat featureMat;   //839*156
    featureMat.load("LBPPCANorFeatures.csv",csv_ascii);  
    arma::mat x0;          //1*156
    x0.load("test.csv",csv_ascii);
    //在读取的两个csv文件中，一个样本的特征占一行，所以需要取转置
    arma::mat A1 = featureMat.t(); 
    arma::mat y = x0.t();

    std::cout<<"Start of KL1p compressed-sensing example."<<std::endl;
    std::cout<<"Try to determine a sparse vector x "<<std::endl;
    std::cout<<"from an underdetermined set of linear measurements y=A*x, "<<std::endl;
    std::cout<<"where A is a random gaussian i.i.d sensing matrix."<<std::endl;

    klab::UInt32 n = A1.n_rows;     // Size of the original signal x0.
    klab::DoubleReal alpha = 0.5;   // Ratio of the cs-measurements.
    klab::DoubleReal rho = 0.1;     // Ratio of the sparsity of the signal x0.
    klab::UInt32 m = klab::UInt32(alpha*n); // Number of cs-measurements.
    klab::UInt32 k = klab::UInt32(rho*n);   // Sparsity of the signal x0 (number of non-zero elements).
    klab::UInt64 seed = 0;                  // Seed used for random number generation (0 if regenerate random numbers on each launch).
    bool bWrite = true;                 // Write signals to files ?


    // Display signal informations.
    std::cout<<"=============================="<<std::endl;
    std::cout<<"N="<<n<<" (signal size)"<<std::endl;
    std::cout<<"M="<<m<<"="<<std::setprecision(5)<<(alpha*100.0)<<"% (number of measurements)"<<std::endl;
    std::cout<<"K="<<k<<"="<<std::setprecision(5)<<(rho*100.0)<<"% (signal sparsity)"<<std::endl;
    std::cout<<"=============================="<<std::endl;

    kl1p::TMatrixOperator<klab::DoubleReal> * matrix = new kl1p::TMatrixOperator<klab::DoubleReal>(A1);
    klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal, klab::DoubleReal> > * A2 =new klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal, klab::DoubleReal> >(matrix);
    klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal, klab::DoubleReal> >  A = *A2;

    klab::DoubleReal tolerance = 1e-3;  // Tolerance of the solution.
    arma::Col<klab::DoubleReal> x;      // Will contain the solution of the reconstruction.

    klab::KTimer timer;

    // Compute Basis-Pursuit.
    std::cout<<"[BasisPursuit] Start."<<std::endl;
    timer.start();
    kl1p::TBasisPursuitSolver<klab::DoubleReal> bp(tolerance);
    bp.solve(y, A, x);
    timer.stop();
    std::cout<<"[BasisPursuit] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<bp.iterations()<<std::endl;
    if(bWrite)
        x.save("BasisPursuit-Signal.csv",csv_ascii);  // Write solution to a file.

    // Compute OMP.
    std::cout<<"------------------------------"<<std::endl;
    std::cout<<"[OMP] Start."<<std::endl;
    timer.start();
    kl1p::TOMPSolver<klab::DoubleReal> omp(tolerance);
    omp.solve(y, A, k, x);
    timer.stop();
    std::cout<<"[OMP] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<omp.iterations()<<std::endl;
    if(bWrite)
        x.save("OMP-Signal.csv",csv_ascii);  // Write solution to a file.

    // Compute ROMP.
    std::cout<<"------------------------------"<<std::endl;
    std::cout<<"[ROMP] Start."<<std::endl;
    timer.start();
    kl1p::TROMPSolver<klab::DoubleReal> romp(tolerance);
    romp.solve(y, A, k, x);
    timer.stop();
    std::cout<<"[ROMP] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<romp.iterations()<<std::endl;
    if(bWrite)
        x.save("ROMP-Signal.csv",csv_ascii);  // Write solution to a file.

    // Compute CoSaMP.
    std::cout<<"------------------------------"<<std::endl;
    std::cout<<"[CoSaMP] Start."<<std::endl;
    timer.start();
    kl1p::TCoSaMPSolver<klab::DoubleReal> cosamp(tolerance);
    cosamp.solve(y, A, k, x);
    timer.stop();
    std::cout<<"[CoSaMP] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<cosamp.iterations()<<std::endl;
    if(bWrite)
        x.save("CoSaMP-Signal.csv",csv_ascii);  // Write solution to a file.

    // Compute Subspace-Pursuit.
    std::cout<<"------------------------------"<<std::endl;
    std::cout<<"[SubspacePursuit] Start."<<std::endl;
    timer.start();
    kl1p::TSubspacePursuitSolver<klab::DoubleReal> sp(tolerance);
    sp.solve(y, A, k, x);
    timer.stop();
    std::cout<<"[SubspacePursuit] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<sp.iterations()<<std::endl;
    if(bWrite)
        x.save("SubspacePursuit-Signal.csv",csv_ascii);     // Write solution to a file.

    // Compute SL0.
    std::cout<<"------------------------------"<<std::endl;
    std::cout<<"[SL0] Start."<<std::endl;
    timer.start();
    kl1p::TSL0Solver<klab::DoubleReal> sl0(tolerance);
    sl0.solve(y, A, x);
    timer.stop();
    std::cout<<"[SL0] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<sl0.iterations()<<std::endl;
    if(bWrite)
        x.save("SL0-Signal.csv",csv_ascii); // Write solution to a file.

    // Compute AMP.
    std::cout<<"------------------------------"<<std::endl;
    std::cout<<"[AMP] Start."<<std::endl;
    timer.start();
    kl1p::TAMPSolver<klab::DoubleReal> amp(tolerance);
    amp.solve(y, A, x);
    timer.stop();
    std::cout<<"[AMP] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
        <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        <<"Iterations="<<amp.iterations()<<std::endl;
    if(bWrite)
        x.save("AMP-Signal.csv",csv_ascii); // Write solution to a file

    system("pause");
    return 0;
}
#endif
