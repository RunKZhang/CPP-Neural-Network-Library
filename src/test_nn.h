#ifndef TEST_NNH
#define TEST_NNH
#include<iostream>
#include "fully_connected.h"
#include "mse.h"
#include<Eigen/Dense>
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
using namespace std;
using namespace autodiff;

void test_proportion_func(auto p_layer_output, auto p_layer1, VectorXd v_new, VectorXd v_out_new)
{
    for(int i=0;i<200;i++)
    {
       v_new = VectorXd::Random(1);
       v_out_new = 4* v_new;
       p_layer_output ->set_output_label(v_out_new);
       cout<<"epoch:"<<i<<", ";
       p_layer1->activation(v_new);
       cout<<"The loss is:"<<p_layer_output->get_loss_value();
       cout<<" The initial value is:"<<v_new<<" The final value is:"<<p_layer_output->get_current_layer_vector()<<endl;
    }

}
void test_polynomial_func(auto p_layer_output, auto p_layer1, VectorXd v_new, VectorXd v_out_new)
{
    for(int i=0;i<400;i++)
    {
        v_new = VectorXd::Random(1);
        v_out_new = 2*(v_new*v_new);
        p_layer_output ->set_output_label(v_out_new);
        cout<<"epoch:"<<i<<", ";
        p_layer1->activation(v_new);
        cout<<"The loss is:"<<p_layer_output->get_loss_value();
        cout<<" The initial value is:"<<v_new<<" The final value is:"<<p_layer_output->get_current_layer_vector()<<endl;
    }
}
void test_step_func(auto p_layer_output, auto p_layer1, VectorXd v_new, VectorXd v_out_new)
{
    for(int i=0;i<400;i++)
    {
        v_new = VectorXd::Random(1);
        v_out_new(0,0) = (v_new.maxCoeff()>0);
        p_layer_output ->set_output_label(v_out_new);
        cout<<"epoch:"<<i<<", ";
        p_layer1->activation(v_new);
        cout<<"The loss is:"<<p_layer_output->get_loss_value();
        cout<<" The initial value is:"<<v_new<<" The final value is:"<<p_layer_output->get_current_layer_vector()<<" true value of v_out_new:"<<v_out_new<<endl;
    }
}
#endif
