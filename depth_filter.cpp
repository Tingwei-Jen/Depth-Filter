#include "depth_filter.h"
#include <boost/math/distributions/normal.hpp>

void Depth_Filter::ComputeTau(const cv::Mat& T12, const cv::Point3f& P, const float& px_error_angle, float& tau)
{

    cv::Point3f t = cv::Point3f(T12.at<float>(0,3), T12.at<float>(1,3), T12.at<float>(2,3));
    cv::Point3f a = P-t;

    float apha = acos( (P.x*t.x + P.y*t.y + P.z*t.z) / (cv::norm(P)*cv::norm(t)) ); 
    float beta = acos( -(a.x*t.x + a.y*t.y + a.z*t.z) / (cv::norm(a)*cv::norm(t)) );
    float beta_plus = beta + px_error_angle;
    float gamma = CV_PI - apha - beta_plus;
    float P_plus_norm = cv::norm(t)*sin(beta_plus)/sin(gamma);

    tau = P_plus_norm - cv::norm(P);

}

void Depth_Filter::UpdateFilter(const float& x, const float& tau2, Seed* seed)
{

    float norm_scale = sqrt(seed->sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;
    boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
    float s2 = 1./(1./seed->sigma2 + 1./tau2);
    float m = s2*(seed->mu/seed->sigma2 + x/tau2);
    float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
    float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
    float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
             + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));
   
    // update parameters
    float mu_new = C1*m+C2*seed->mu;
    seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
    seed->mu = mu_new;
    seed->a = (e-f)/(f-e/f);
    seed->b = seed->a*(1.0f-f)/f;

}
