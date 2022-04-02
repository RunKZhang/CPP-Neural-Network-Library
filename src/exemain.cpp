#include<iostream>
#include "fully_connected.h"
#include "mse.h"
#include<Eigen/Dense>
#include<autodiff/reverse/var.hpp>
#include<autodiff/reverse/var/eigen.hpp>
#include "test_nn.h"
#include "activ_func.h"
#include<fstream>
#include<vector>
#include "process_dataset.h"
using namespace std;
using namespace autodiff;

int main()
{
    auto p_layer_output =make_shared<Mse<Softmax>>(3);
    //auto p_layer3 =make_shared<FullConnect<Tanh>>(64,dynamic_pointer_cast<NeuronBase>(p_layer_output));
    auto p_layer2 =make_shared<FullConnect<Tanh>>(64,dynamic_pointer_cast<NeuronBase>(p_layer_output));
    auto p_layer1 =make_shared<FullConnect<linear>>(4,dynamic_pointer_cast<NeuronBase>(p_layer2));


    vector<vector<string>> outer_content;
    outer_content = get_dataset();
    MatrixXd input_matrix = transfer_dataset(outer_content);
    MatrixXd label_matrix = set_one_hot_label(outer_content);
    double total_loss;
    double accuracy=0;
    int i = 0;
    p_layer_output->set_acc();
    for(int j=0;j<5960;j++)
    {


        i = (i+23)%149;
        p_layer_output ->set_output_label(label_matrix.col(i));
        p_layer1 -> activation(input_matrix.col(i));

        if(j%149 == 0)
        {
            accuracy = p_layer_output->get_acc()/149*100;
            cout<<"epoch:"<<(j/149)<<" The accuracy is:"<<accuracy<<"%"<<" "<<"The loss value is:"<<p_layer_output->get_loss_value()<<endl;
            p_layer_output->set_acc();
        }
    }
    return 0;
}
