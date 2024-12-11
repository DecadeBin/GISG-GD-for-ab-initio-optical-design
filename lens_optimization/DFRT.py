# 最新版函数库
import gc
import time
import torch.nn.functional as F
import psutil
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import pandas as pd
# import mamba_ssm
from torchvision import transforms
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

torch.set_default_dtype(torch.float64)
torch.cuda.set_device(0)
efl_scale = 1



# def ray_trace_newton(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o, alpha, beta, gama, k, c, rn_in, rn_out, omit,
#                      power_para, th_labal=None,cal_op = False,name='None'):
#     # 给定下一个面的坐标系原点相对当前坐标系原点的位置，将当前坐标系光线，追迹到下一个表面，并得到折射后的光线的信息
#     # rn_in现在是[3,1],需要变成[1,3,1]
#
#     k = k + 1
#
#     # 首先需要一个坐标系相对位置x_o,y_o,z_o,以及旋转角度alpha,beta,gama
#     # 函数输入还应该有光线在当前坐标系的位置和方向信息x_out,y_out,z_out,l,m,n
#     # 使用coordinate_change_with_rotation进行坐标系变换
#
#     x_in, y_in, z_in, l_in, m_in, n_in = coordinate_change_with_rotation(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o,
#                                                                          alpha, beta, gama)
#
#     # with torch.no_grad():
#     # 得到新坐标系下的入射光线后，使用计算ray_with_surf_by_newton光线和曲面的交点，因此应该在函数输入里面加上去曲面的信息k,c
#     x_intersect, y_intersect, z_intersect, opl, loss_intersect = ray_with_surf_by_newton(k, c, x_in, y_in, z_in, l_in,
#                                                                                          m_in, n_in,
#                                                                                          rn_in, omit, power_para,cal_op)
#
#     # 偶尔增加内存，将求交点的几个地方改成.data这里就没问题了
#     # 得到新坐标系下的交点后，需要计算交点法线refraction_by_newton，从而计算折射后光线方向，因此还需要再输入加上折射面前后的折射率
#
#     l_out, m_out, n_out, refract_loss, loss_curve = refraction_by_newton(k, c, x_intersect, y_intersect, z_intersect,
#                                                                          l_in, m_in, n_in, rn_in, rn_out, omit,
#                                                                          power_para)
#
#     # 该函数返回光传播+光折射后，在当前坐标系的交点和出射光线方向,因此要返回x_intersect,y_intersect,z_intersect和l_out,m_out,n_out,同时将这一段的光程opl返回
#     # 对于手机镜头，空气0.02，玻璃0.1
#     if th_labal == 3:
#         min_z = -100 / efl_scale
#     elif th_labal == 0:
#         min_z = 0.1 / efl_scale  # 空气
#     else:
#         min_z = 0.2 / efl_scale  # 玻璃
#
#     delta_z = torch.min(torch.min(z_intersect - z_in, dim=-1).values, dim=-1, keepdim=True).values
#     delta_z = torch.where(delta_z < min_z / efl_scale, min_z / efl_scale - delta_z, 0.)
#     # delta_z = torch.sum(delta_z, dim=-1, keepdim=True)
#
#     return x_intersect, y_intersect, z_intersect, l_out, m_out, n_out, opl, delta_z, refract_loss, loss_curve

# 光学设计函数
def ray_trace_newton(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o, alpha_o, beta_o, gama_o, k, c, rn_in, rn_out, omit,
                     power_para, th_labal=None,cal_op = False,edge = [0.1,0.2,0.15] ):
    # 给定下一个面的坐标系原点相对当前坐标系原点的位置，将当前坐标系光线，追迹到下一个表面，并得到折射后的光线的信息
    # alpha_o是下一表面的信息，alpha_in是前一表面的信息，默认alpha_in为0，
    # rn_in现在是[3,1],需要变成[1,3,1]

    k = k + 1

    # 首先需要一个坐标系相对位置x_o,y_o,z_o,以及旋转角度alpha,beta,gama
    # 函数输入还应该有光线在当前坐标系的位置和方向信息x_out,y_out,z_out,l,m,n
    # 使用coordinate_change_with_rotation进行坐标系变换
    rotate_matrix = rotation_matrix_inverse(alpha_o, beta_o, gama_o)
    # 输入的数据应当是以第一个面（光阑）的坐标系为参考，z除外
    # 因此要先经过一个坐标变换，变换到局部坐标系
    x_out, y_out, z_out,  l, m, n = coordinate_change_with_rotation(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o,
                                                                         rotate_matrix)

    # with torch.no_grad():
    # 得到新坐标系下的入射光线后，使用计算ray_with_surf_by_newton光线和曲面的交点，因此应该在函数输入里面加上去曲面的信息k,c
    x_out, y_out, z_intersect, opl, loss_intersect = ray_with_surf_by_newton(k, c, x_out, y_out, z_out,  l, m, n,
                                                                                         rn_in, omit, power_para,cal_op)

    # 得到新坐标系下的交点后，需要计算交点法线refraction_by_newton，从而计算折射后光线方向，因此还需要再输入加上折射面前后的折射率
    l, m, n, refract_loss, loss_curve = refraction_by_newton(k, c, x_out, y_out, z_intersect,
                                                                         l, m, n, rn_in, rn_out, omit,
                                                                         power_para)

    # if alpha_tensor==True:
    rotate_matrix = torch.inverse(rotate_matrix)
    # rotate_matrix2 = rotation_matrix_inverse(-alpha_o, -beta_o, gama_o)

    x_out, y_out, z_intersect, l, m, n  = coordinate_change_with_rotation(x_out, y_out, z_intersect, l, m, n,-x_o, -y_o, 0,
                                                                         rotate_matrix,local2global=True)

    # 光线追迹结束后，x,y,l,m,n要变回全局坐标系。因此，x,y要乘以旋转矩阵的逆，lmn也是

    # 该函数返回光传播+光折射后，在当前坐标系的交点和出射光线方向,因此要返回x_intersect,y_intersect,z_intersect和l_out,m_out,n_out,同时将这一段的光程opl返回
    # 对于手机镜头，空气0.02，玻璃0.1
    if th_labal == 3:
        min_z = edge[2]
    elif th_labal == 0:
        min_z = edge[0]   # 空气
    else:
        min_z = edge[1]   # 玻璃

    delta_z = torch.min(torch.min(z_intersect - z_out, dim=-1).values, dim=-1, keepdim=True).values
    delta_z = torch.where(delta_z < min_z / efl_scale, min_z / efl_scale - delta_z, 0.)
    # delta_z = torch.sum(delta_z, dim=-1, keepdim=True)

    return  x_out, y_out, z_intersect,  l, m, n , opl, delta_z, refract_loss, loss_curve


def ray_trace_newton_only_point(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o, alpha_o, beta_o, gama_o, k, c, rn_in, omit,
                     power_para, th_labal=None,cal_op = False):
    # 给定下一个面的坐标系原点相对当前坐标系原点的位置，将当前坐标系光线，追迹到下一个表面，并得到折射后的光线的信息
    # rn_in现在是[3,1],需要变成[1,3,1]
    k = k + 1

    # 首先需要一个坐标系相对位置x_o,y_o,z_o,以及旋转角度alpha,beta,gama
    # 函数输入还应该有光线在当前坐标系的位置和方向信息x_out,y_out,z_out,l,m,n
    # 使用coordinate_change_with_rotation进行坐标系变换
    rotate_matrix = rotation_matrix_inverse(alpha_o, beta_o, gama_o)
    # 输入的数据应当是以第一个面（光阑）的坐标系为参考，z除外
    # 因此要先经过一个坐标变换，变换到局部坐标系
    x_in, y_in, z_in, l_in, m_in, n_in = coordinate_change_with_rotation(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o,
                                                                         rotate_matrix)

    # with torch.no_grad():
    # 得到新坐标系下的入射光线后，使用计算ray_with_surf_by_newton光线和曲面的交点，因此应该在函数输入里面加上去曲面的信息k,c
    x_intersect, y_intersect, z_intersect, opl, loss_intersect = ray_with_surf_by_newton(k, c, x_in, y_in, z_in, l_in,
                                                                                         m_in, n_in,
                                                                                         rn_in, omit, power_para,
                                                                                         cal_op)
    x_intersect, y_intersect, z_intersect, l_out, m_out, n_out  = coordinate_change_with_rotation(x_intersect, y_intersect, z_intersect, l_in, m_in, n_in,-x_o, -y_o, 0,
                                                                         rotate_matrix)
    # 偶尔增加内存，将求交点的几个地方改成.data这里就没问题了

    return x_intersect, y_intersect, z_intersect, opl


def grid_data_of_staring_point_para(stop_half_apture, field_y=[0, 6.67, 10], sample_delta=0.1, field_x=None, obj=-10e7,
                                    field_type=0, cal_simul=False):
    # cal_simul决定时候在光线追迹的同时计算Efl
    if field_x != None:
        field_y = torch.tensor(field_y)
        field_x = torch.tensor(field_x)
        field = torch.meshgrid(field_x, field_y)
        field_x = field[0].contiguous().view(-1, 1)
        field_y = field[1].contiguous().view(-1, 1)
    else:
        field_y = torch.tensor(field_y).unsqueeze(1)
        field_x = torch.zeros(field_y.shape)
    field_num = field_y.shape[0]
    if field_type == 0:
        x0_object = obj * torch.tan(field_x / 180 * math.pi)  # 0.
        z0_object = obj
        y0_object = obj * torch.tan(field_y / 180 * math.pi)
    else:
        x0_object = field_x
        z0_object = obj
        y0_object = field_y
    # 物电位置x上到下为负到正，y从左到右为负到正

    xgrid = torch.linspace(stop_half_apture, -stop_half_apture, sample_delta)
    ygrid = torch.linspace(stop_half_apture, -stop_half_apture, sample_delta)
    # 采样点栅格，从上到下x从正到负，从左到右为y从正到负
    mesh_xy = torch.meshgrid(xgrid, ygrid)
    xgrid = mesh_xy[0].reshape(1, -1)
    ygrid = mesh_xy[1].reshape(1, -1)

    radius = (xgrid ** 2 + ygrid ** 2)
    mask = radius <= (stop_half_apture * stop_half_apture)
    xgrid = xgrid[mask].unsqueeze(0).repeat_interleave(field_num, dim=0)
    ygrid = ygrid[mask].unsqueeze(0).repeat_interleave(field_num, dim=0)
    zgrid = torch.zeros(ygrid.shape)
    length = ((xgrid - x0_object) ** 2 + (ygrid - y0_object) ** 2 + (zgrid - z0_object) ** 2) ** 0.5
    Lgrid = (xgrid - x0_object) / length
    Mgrid = (ygrid - y0_object) / length
    Ngrid = (zgrid - z0_object) / length

    # dis = ((xgrid -x0_object)**2 + (ygrid-y0_object)**2 + (zgrid--z0_object)**2)**2
    # 得到每个视场的等效视场高度
    # x_grid,Lgrid[field_num,nm] y0_object[field_num,1],return opinit[2,5]
    # op_init = entrance_gird2opd_init(xgrid,ygrid,Ngrid,x0_object,y0_object,field_x,field_y,obj=-10e7,ray_aming=False)
    theta_each_field = torch.arctan(torch.sqrt(x0_object ** 2 + y0_object ** 2) / obj)
    theta_each_field = torch.where((x0_object < 0), -theta_each_field, theta_each_field)
    # 计算每个视场点的零光程差线的垂线

    k = -y0_object / x0_object
    d = (k * ygrid - xgrid) / torch.sqrt(k ** 2 + 1)
    op_init = -torch.sin(theta_each_field) * d
    op_init_y = torch.sin(field_y / 180 * math.pi) * ygrid
    op_init_x = torch.sin(field_x / 180 * math.pi) * xgrid
    y0_object = y0_object.repeat(1, Ngrid.shape[-1])
    x0_object = x0_object.repeat(1, Ngrid.shape[-1])

    # if torch.abs(x0_object).sum()!=0:
    op_init = torch.where(x0_object == 0., op_init_y, op_init)
    op_init = torch.where(y0_object == 0., op_init_x, op_init)
    zeros = torch.zeros(1, 1, 1)

    xgrid = xgrid.view(1, 1, -1);
    ygrid = ygrid.view(1, 1, -1);
    zgrid = zgrid.view(1, 1, -1);
    Lgrid = Lgrid.view(1, 1, -1);
    Mgrid = Mgrid.view(1, 1, -1);
    Ngrid = Ngrid.view(1, 1, -1);
    if cal_simul == True:
        xgrid = torch.cat((xgrid, zeros), dim=2)
        ygrid = torch.cat((ygrid, zeros + 0.02), dim=2)
        zgrid = torch.cat((zgrid, zeros), dim=2)
        Lgrid = torch.cat((Lgrid, zeros), dim=2)
        Mgrid = torch.cat((Mgrid, zeros), dim=2)
        Ngrid = torch.cat((Ngrid, zeros + 1), dim=2)

    op_init = op_init.view(1, 1, -1)
    nm = xgrid.shape[-1]
    # 生成光阑处采样点时，xyzlmn均加入一个用来算efl的数
    return xgrid.cuda(), ygrid.cuda(), zgrid.cuda(), Lgrid.cuda(), Mgrid.cuda(), Ngrid.cuda(), nm, sample_delta   , op_init.cuda()


def grid_data_of_staring_point_para_only_chief(stop_half_apture, field_y=[0, 6.67, 10], sample_delta=0.1, field_x=None, obj=-10e7,
                                    field_type=0, cal_simul=False):
    # cal_simul决定时候在光线追迹的同时计算Efl
    if field_x != None:
        field_y = torch.tensor(field_y)
        field_x = torch.tensor(field_x)
        field = torch.meshgrid(field_x, field_y)
        field_x = field[0].view(-1, 1)
        field_y = field[1].view(-1, 1)
    else:
        field_y = torch.tensor(field_y).unsqueeze(1)
        field_x = torch.zeros(field_y.shape)
    field_num = field_y.shape[0]
    if field_type == 0:
        x0_object = obj * torch.tan(field_x / 180 * math.pi)  # 0.
        z0_object = obj
        y0_object = obj * torch.tan(field_y / 180 * math.pi)
    else:
        x0_object = field_x
        z0_object = obj
        y0_object = field_y

    xgrid = torch.linspace(-stop_half_apture, stop_half_apture, sample_delta)
    ygrid = torch.linspace(-stop_half_apture, stop_half_apture, sample_delta)

    mesh_xy = torch.meshgrid(xgrid, ygrid)
    xgrid = mesh_xy[0].view(1, -1)
    ygrid = mesh_xy[1].view(1, -1)

    radius = (xgrid ** 2 + ygrid ** 2)
    mask = radius < (stop_half_apture * stop_half_apture)
    xgrid = xgrid[mask].unsqueeze(0).repeat_interleave(field_num, dim=0)
    ygrid = ygrid[mask].unsqueeze(0).repeat_interleave(field_num, dim=0)
    zgrid = torch.zeros(ygrid.shape)
    length = ((xgrid - x0_object) ** 2 + (ygrid - y0_object) ** 2 + (zgrid - z0_object) ** 2) ** 0.5
    Lgrid = (xgrid - x0_object) / length
    Mgrid = (ygrid - y0_object) / length
    Ngrid = (zgrid - z0_object) / length

    # dis = ((xgrid -x0_object)**2 + (ygrid-y0_object)**2 + (zgrid--z0_object)**2)**2
    # 得到每个视场的等效视场高度
    theta_each_field =torch.arctan(torch.sqrt(x0_object**2 + y0_object**2 ) / obj)
    # 计算每个视场点的零光程差线的垂线
    k = - x0_object/y0_object
    d = (k * xgrid - ygrid) / torch.sqrt(k**2+1)
    op_init_nonan = torch.sin(theta_each_field ) * d
    op_init = torch.sin(field_x / 180 * math.pi) * xgrid
    op_init = torch.where(torch.isnan(op_init_nonan),op_init,op_init_nonan)
    zeros = torch.zeros(1, 1, 1)

    xgrid = xgrid.view(1, 1, -1);
    ygrid = ygrid.view(1, 1, -1);
    zgrid = zgrid.view(1, 1, -1);
    Lgrid = Lgrid.view(1, 1, -1);
    Mgrid = Mgrid.view(1, 1, -1);
    Ngrid = Ngrid.view(1, 1, -1);
    if cal_simul == True:
        xgrid = torch.cat((xgrid, zeros), dim=2)
        ygrid = torch.cat((ygrid, zeros + 0.02), dim=2)
        zgrid = torch.cat((zgrid, zeros), dim=2)
        Lgrid = torch.cat((Lgrid, zeros), dim=2)
        Mgrid = torch.cat((Mgrid, zeros), dim=2)
        Ngrid = torch.cat((Ngrid, zeros + 1), dim=2)

    op_init = op_init.view(1, 1, -1)
    nm = xgrid.shape[-1] - 1
    # 生成光阑处采样点时，xyzlmn均加入一个用来算efl的数
    return xgrid.cuda(), ygrid.cuda(), zgrid.cuda(), Lgrid.cuda(), Mgrid.cuda(), Ngrid.cuda(), nm, sample_delta   , op_init.cuda()

def generate_field_and_ideal(field_y,target_efl,field_type,field_x=None):

    if field_x != None:
        field_y = torch.tensor(field_y)
        field_x = torch.tensor(field_x)
        field = torch.meshgrid(field_x, field_y)
        field_x = field[0].view(-1).tolist()
        field_y = field[1].view(-1).tolist()



    if field_type==0:
        y_field_ideal = target_efl * torch.tan(torch.tensor([field_y]) / 180 * math.pi).cuda()
    if field_type==1:
        y_field_ideal = (torch.tensor(field_y)*target_efl).unsqueeze(0).cuda()
    field_num = y_field_ideal.shape[-1]
    if field_x != None:
        if field_type == 0:
            x_field_ideal = target_efl * torch.tan(torch.tensor([field_x]) / 180 * math.pi).cuda()
        if field_type == 1:
            x_field_ideal = (torch.tensor(field_x) * target_efl).unsqueeze(0).cuda()
    else:
        x_field_ideal = 0
        field_x=0
    return y_field_ideal,x_field_ideal,field_num


zero_one = torch.tensor([0.]).cuda()

def rotation_matrix_inverse(alpha, beta, gama):
    # alpha[N,1]
    gama = zero_one
    N = alpha.shape[0]
    if len(alpha.shape)==4:

        R = torch.zeros([N, beta.shape[1], beta.shape[2],3,3]).cuda()
        R[:,:,: ,0, 0] = (torch.cos(alpha) * torch.cos(gama) + torch.sin(alpha) * torch.sin(beta) * torch.sin(gama))[:,:,:,0]
        R[:,:,:, 0, 1] = (-  torch.cos(beta) * torch.sin(gama))[:,:,:,0]
        R[:,:,:, 0, 2] = (-  torch.sin(alpha) * torch.cos(gama) + torch.cos(alpha) * torch.sin(beta) * torch.sin(gama))[:,:,:,0]

        R[:,:,:, 1, 0] = (torch.cos(alpha) * torch.sin(gama) - torch.sin(alpha) * torch.sin(
            beta) * torch.cos(gama))[:,:,:,0]
        R[:,:,:, 1, 1] = (torch.cos(beta) * torch.cos(gama))[:,:,:,0]
        R[:,:,:, 1, 2] = (-  torch.sin(alpha) * torch.sin(gama) - torch.cos(
            alpha) * torch.sin(beta) * torch.cos(gama))[:,:,:,0]
        R[:,:,:, 2, 0] = (torch.sin(alpha) * torch.cos(beta))[:,:,:,0]
        R[:,:,:, 2, 1] = (torch.sin(beta))[:,:,:,0]
        R[:,:,:, 2, 2] = (torch.cos(alpha) * torch.cos(beta))[:,:,:,0]
    else:
        R = torch.zeros([N,3,3]).cuda()
        R[:,0,0] = (torch.cos(alpha) * torch.cos(gama) + torch.sin(alpha) * torch.sin(beta) * torch.sin(gama))[:,0,0]
        R[:, 0, 1] = (-  torch.cos(beta) * torch.sin(gama))[:,0,0]
        R[:, 0, 2] = (-  torch.sin(alpha) * torch.cos(gama) + torch.cos(alpha) * torch.sin(beta) * torch.sin(gama))[:,0,0]

        R[:, 1, 0] = (torch.cos(alpha) * torch.sin(gama) - torch.sin(alpha) * torch.sin(
            beta) * torch.cos(gama))[:,0,0]
        R[:, 1, 1] = (torch.cos(beta) * torch.cos(gama))[:,0,0]
        R[:, 1, 2] = (-  torch.sin(alpha) * torch.sin(gama) - torch.cos(
            alpha) * torch.sin(beta) * torch.cos(gama))[:,0,0]
        R[:, 2, 0] = (torch.sin(alpha) * torch.cos(beta))[:,0,0]
        R[:, 2, 1] = (torch.sin(beta))[:,0,0]
        R[:, 2, 2] = (torch.cos(alpha) * torch.cos(beta))[:,0,0]

    # R11 = torch.cos(alpha) * torch.cos(gama) + torch.sin(alpha) * torch.sin(
    #     beta) * torch.sin(gama)
    # R12 = -  torch.cos(beta) * torch.sin(gama)
    # R13 = -  torch.sin(alpha) * torch.cos(gama) + torch.cos(
    #     alpha) * torch.sin(beta) * torch.sin(gama)
    # R21 = torch.cos(alpha) * torch.sin(gama) - torch.sin(alpha) * torch.sin(
    #     beta) * torch.cos(gama)
    # R22 = torch.cos(beta) * torch.cos(gama)
    # R23 = -  torch.sin(alpha) * torch.sin(gama) - torch.cos(
    #     alpha) * torch.sin(beta) * torch.cos(gama)
    # R31 = torch.sin(alpha) * torch.cos(beta)
    # R32 = torch.sin(beta)
    # R33 = torch.cos(alpha) * torch.cos(beta)

    return R


# 1、坐标变换函数
def coordinate_change_with_rotation(x_out, y_out, z_out, l, m, n, x_o, y_o, z_o,rotate_matrix, local2global=False):
    # beta对应acos(n) alpha对应acos(l)  gama对应acos(m)
    # 坐标旋转：将初始坐标系中的(x_out,y_out,z_out)和方向信息,O(x_o,y_o,z_o)[新的坐标系的原点在原坐标系的位置]。转换到新的坐标系得到点坐标[x_in,y_in,z_in]
    # x_out等应该是tensor，x_o等可以是数字
    # _out指的是从前一表面出来的光线，_in指的是下一表面要进入的光线
    if local2global==False:
        x_out = x_out -  x_o
        y_out = y_out - y_o
        z_out = z_out - z_o
    if len(rotate_matrix.shape)==5:
        R11 = rotate_matrix[:,:,:,0:1,0]
        R12 = rotate_matrix[:,:,:,0:1,1]
        R13 = rotate_matrix[:,:,:,0:1,2]
        R21 = rotate_matrix[:,:,:,1:2,0]
        R22 = rotate_matrix[:,:,:,1:2,1]
        R23 = rotate_matrix[:,:,:,1:2,2]
        R31 = rotate_matrix[:,:,:,2:3,0]
        R32 = rotate_matrix[:,:,:,2:3,1]
        R33 = rotate_matrix[:,:,:,2:3,2]
    else:
        R11 = rotate_matrix[:, 0:1, 0:1]
        R12 = rotate_matrix[:, 0:1, 1:2]
        R13 = rotate_matrix[:, 0:1, 2:3]
        R21 = rotate_matrix[:, 1:2, 0:1]
        R22 = rotate_matrix[:, 1:2, 1:2]
        R23 = rotate_matrix[:, 1:2, 2:3]
        R31 = rotate_matrix[:, 2:3, 0:1]
        R32 = rotate_matrix[:, 2:3, 1:2]
        R33 = rotate_matrix[:, 2:3, 2:3]

    #  旋转矩阵R
    x_in = R11 * x_out + R12 * y_out + R13 * z_out
    y_in = R21 * x_out + R22 * y_out + R23 * z_out
    z_in = R31 * x_out + R32 * y_out + R33 * z_out

    if local2global==True:
        x_in = x_in -  x_o
        y_in = y_in - y_o
        z_in = z_in - z_o

    l_in = R11 * l + R12 * m + R13 * n
    m_in = R21 * l + R22 * m + R23 * n
    n_in = R31 * l + R32 * m + R33 * n
    return x_in, y_in, z_in, l_in, m_in, n_in


# 2、用牛顿法计算直线和曲面的交点
def ray_with_surf_by_newton(k, c, x_in, y_in, z_in, l, m, n, rn, omit, power_para, cal_opl=False):
    # 该函数接收前一表面的光线的起始点和方向(x_in, y_in,z_in), 要求已经通过坐标变换变成相对于当前去曲面的坐标系中，即当前曲面的切平面在z = 0处；
    '''
    测试案例
    x_in = torch.tensor([-2.4])
    y_in = torch.tensor([0])
    z_in = torch.tensor([0])
    l = 0
    m = 0
    n = 1
    k = 1
    c = 1 / 21.48138
    rn = 1
    x_intersect, y_intersect, z_intersect, opl = ray_with_surf_by_newton(k, c, x_in, y_in, z_in, l, m, n, rn)
    '''

    # k目前全都设置为1，c = 1 / r，rn是折射率
    s0 = -z_in / n  # 点到切平面的距离
    x0 = x_in + l * s0
    y0 = y_in + m * s0
    z0 = 0# torch.zeros(y0.shape).cuda()# 光线在切平面上的交点。
    # 存在一些情况，以切平面处交点为起点是不行的，需要往弯曲的方向移动一些。
    # 此时：(1 - k * c ** 2 * rou2)<0，s0需要沿着曲率方向在走一段
    error1_mask = ((1 - k * c ** 2 * (x0**2 + y0**2))<0)
    error1_adjust_time = 0
    try:
        c_adjust_so = c.repeat(1, x0.shape[1], x0.shape[2])
    except:
        pass
    if n.shape != s0.shape:
        n = n.repeat(s0.shape[0], s0.shape[1], 1)
    while error1_mask.sum()>0:
        z0 = torch.zeros(y0.shape).cuda()
        error1_adjust_time +=1
        # 需要对错误的点，将s0根据曲率正负移动：

        s0[error1_mask] = s0[error1_mask]  + 1/(error1_adjust_time+1) * (1/c_adjust_so[error1_mask]) / n[error1_mask]
        x0 = x_in + l * s0
        y0 = y_in + m * s0
        z0[error1_mask] =  1/(error1_adjust_time+1) * (1/c_adjust_so[error1_mask])
        error1_mask = ((1 - k * c ** 2 * (x0 ** 2 + y0 ** 2)) < 0)
        if error1_adjust_time>5:
            break

    ###按照论文的说法，先追迹到切平面，即得到s0和(x0, y0, z0)，然后从该点开始，追迹到曲面，距离为s，s为通过迭代法计算的变量

    # 这里是避免对于曲率为负的面，如果以曲面切线为起点，会导致rou2大于曲面最大值的情况
    s_next = 0.
    N = 10
    ########## 这一段和曲面类型没有关系↑

    ################################################################################################################################################
    # with torch.no_grad():
    for i in range(N):
        # 曲面交点的坐标用s0和(x0, y0, z0)表示为
        x1 = x0 + l * s_next
        y1 = y0 + m * s_next
        z1 = z0 + n * s_next

        ############这一段需要根据曲面类型设置F(x1, y1, z1)和F'(x1,y1,z1) ↓
        ### 先写出曲面关于X，Y，Z的导数, 参考论文"General Ray-Tracing Procedure"第5页公式16的表达式，省略后面的加项

        F, F_d = newton_iter_multi_ray_with_rotation_symmetric(k, c, x1, y1, z1, l, m, n, omit, power_para)
        # 步长方向应该和F相同

        step = F / F_d
        if torch.max((torch.abs(step))) < (10 ** (-6)):
            s_next = (s_next - step)
            break

        # 解决光线和曲面平行时，导数太小导致步长太大引起的牛顿迭代失败，如果下一步更新会使得求不出F，则将step替换成F
        s_next = (s_next - step)
        # if i == N - 1:
        #     print('求交点达到最大迭代次数')
        # s = s_next#.data

    x_intersect = x0 + l * s_next
    y_intersect = y0 + m * s_next
    z_intersect = z0 + n * s_next
    # 通过判断一个点F是否为零判断是否这个点找没找到交点，没有交点需要有损失，
    # F, F_d = newton_iter_multi_ray_with_rotation_symmetric(k, c, x_intersect, y_intersect, z_intersect, l, m, n, omit,power_para)
    loss_intersect = 0.
    if cal_opl == True:
        try:
            opl = (s0 + s_next) * rn
        except:
            opl = (s0 + s_next) * rn.unsqueeze(-1)
    else:
        opl = torch.tensor([0])  #

    return x_intersect, y_intersect, z_intersect, opl, loss_intersect


# 2.1 曲面的表达式


def newton_iter_multi_ray_with_rotation_symmetric(k, c, x, y, z, l, m, n, omit, power_para):
    # 返回旋转对称曲面(参数为k, c)的F(x, y, z)和F'(x,y,z).因为导数最终是关于s的，所以还需要光线的方向。x,y,z可以是[1,num]的行向量y
    # omit设置为0, 则忽略旋转对称曲面一般表达式后面的加项，如果为1，则忽略，从而使用另一种表达式。
    # 如果k = 1, c = 1 / R, 则此时曲面为半径为R的球面，球心在z = R处
    # power_para控制非球面系数，[曲面数量，高次项数]

    rou2 = x ** 2 + y ** 2
    if omit == 0:
        power_num = power_para.shape[-1]
        rou_t = rou2.unsqueeze(-2).repeat_interleave(power_num, dim=-2)  # rout[N,,pnum,nm]
        pp = (power_para.transpose(1, 2)).unsqueeze(-3)
        level = torch.zeros((1, 1, power_num, 1)).cuda()
        # rou2[N,nm] power_para[N,1,num] i[1,num,1]
        for j in range(power_num):
            i = j + 2
            level[0, 0, j, 0] = i
        E_high_power = 2 * level * pp * (rou_t ** (level - 1))
        E_high_power = torch.sum(E_high_power, dim=-2)
        E = c / (1 - k * c ** 2 * rou2) ** 0.5
        E = E + E_high_power  # 问题出现在这个E_high_power,有他就会报错
        Fx = -x * E
        Fy = -y * E
        Fz = 1
        high_power_value = pp * (rou_t ** level)
        high_power_value = torch.sum(high_power_value, dim=-2)
        # 这里为了避免出现复数，要求rou < 1 / sqrt(k * c)
        F = z - (c * rou2) / (1 + (1 - k * c ** 2 * rou2) ** 0.5) - high_power_value  # F(X, Y, Z)
        F_d = Fx * l + Fy * m + Fz * n
    else:
        ### 偶尔增加内存占用
        E = c / (1 - k * c ** 2 * rou2) ** 0.5  # 验证下两个非球面的表达式都可以训练
        Fx = -x * E
        Fy = -y * E

        ###
        Fz = 1
        F = z - (c * rou2) / (1 + (1 - k * c ** 2 * rou2) ** 0.5)
        F_d = Fx * l + Fy * m + Fz * n

    return F, F_d


def refraction_by_newton(k, c, x, y, z, l, m, n, rn_in, rn_out, omit, power_para, grad_max=0.):
    rou2 = x ** 2 + y ** 2
    if omit == 0:
        power_num = power_para.shape[-1]
        rou_t = rou2.unsqueeze(-2).repeat_interleave(power_num, dim=-2)  # rout[N,,pnum,nm]
        pp = (power_para.transpose(1, 2)).unsqueeze(-3)
        level = torch.zeros((1, 1, power_num, 1)).cuda()
        # rou2[N,nm] power_para[N,1,num] i[1,num,1]
        for j in range(power_num):
            i = j + 2
            level[0, 0, j, 0] = i
        E_high_power = 2 * level * pp * (rou_t ** (level - 1))
        E_high_power = torch.sum(E_high_power, dim=-2)
        E = c / (1 - k * c ** 2 * rou2) ** 0.5
        E = E + E_high_power  # 问题出现在这个E_high_power,有他就会报错
        L = -x * E
        M = -y * E
        N = 1
    else:

        E = c / (1 - k * c ** 2 * rou2) ** 0.5  # q去掉那三个.data的注释，这里就内存不增加
        L = -x * E
        M = -y * E
        N = 1
        # L = -c * x
        # M = -c * y
        # N = 1 - k * c * z
    length = torch.sqrt(L ** 2 + M ** 2 + N ** 2)
    L = L / length
    M = M / length
    N = N / length  # 法线的方向，这个余弦值不能太小，否则说明曲面太弯了

    n_nonan = torch.where(torch.isnan(N), 200., N)
    n_nonan = (torch.min(n_nonan,dim=-1)).values
    angle_limit = torch.where(n_nonan < 0.7, 0.7-n_nonan, 0.)
    angle_limit = torch.sum(angle_limit,dim=-1,keepdim=True)
    tao = rn_in / rn_out  # [1,3,1]
    cos_phi = l * L + m * M + n * N
    # ir_limit = 1.15 * (1 - (rn_out / rn_in) ** 2)**0.5
    if torch.max(rn_out) < torch.max(rn_in):
        tir_limit = 1.1 * (1 - (rn_out / rn_in) ** 2) ** 0.5
        phi_loss = torch.min(cos_phi, dim=-1, keepdim=True).values
        phi_loss = torch.where(phi_loss < tir_limit, tir_limit - phi_loss, 0.)
        phi_loss = torch.sum(phi_loss, dim=1)
        # phi_loss = torch.sum(phi_loss,dim=-1,keepdim=True)
        # if torch.max(phi_loss) > 0:
        #     print('全反射限制生效')
        #     print(torch.max(phi_loss))
    else:
        phi_loss = torch.min(cos_phi, dim=-1, keepdim=True).values
        phi_loss = torch.where(phi_loss < 0.7, 0.7 - phi_loss, 0.)
        phi_loss = torch.sum(phi_loss, dim=1)
        # phi_loss = torch.sum(phi_loss, dim=-1, keepdim=True)
        # if torch.max(phi_loss) > 0:
        #     print('无交点限制生效')
        #     print(torch.max(phi_loss))
        # phi_loss = 0.
    cos_phi_1 = torch.sqrt(1 - (tao ** 2) * (1 - cos_phi ** 2))

    # if torch.isnan(cos_phi_1).any():
    #     print('发生全反射')
    temp = tao * cos_phi - cos_phi_1
    l_out = tao * l - temp * L
    m_out = tao * m - temp * M
    n_out = tao * n - temp * N
    return l_out, m_out, n_out, 0., phi_loss+angle_limit


def cal_mlzi_spot_size(x, y, field_num=3):
    batch = x.shape[0]
    nm = int(x.shape[-1] / field_num)
    wave_num = x.shape[1]
    spotsize = torch.zeros(batch, field_num).cuda()
    avg_y_list = spotsize.clone()
    avg_x_list = spotsize.clone()# torch.zeros(batch, field_num).cuda()
    y_real_mw = torch.zeros(batch,x.shape[1], field_num).cuda()
    x_real_mw = torch.zeros(batch, x.shape[1], field_num).cuda()
    center = nm//2
    for i in range(field_num):
        spot_x = x[:, :, nm * i:nm * (i + 1)]
        spot_y = y[:, :, nm * i:nm * (i + 1)]
        sum_x = torch.sum(torch.sum(spot_x, dim=-1), dim=-1, keepdim=True)
        sum_y = torch.sum(torch.sum(spot_y, dim=-1), dim=-1, keepdim=True)
        avg_x = sum_x / nm / wave_num  # [batch,1]
        avg_y = sum_y / nm / wave_num  # avg_y_list[:,i] = avg_y[:,0]
        avg_y_list[:, i] = avg_y[:, 0]
        avg_x_list[:, i] = avg_x[:, 0]
        y_real_mw[:,:,i] = spot_y[:,:,center]
        x_real_mw[:, :, i] = spot_x[:, :, center]
        delta_x = torch.sum(torch.sum((spot_x - avg_x.unsqueeze(-1)) ** 2, dim=-1), dim=-1, keepdim=True)  # [batch,1]
        delta_y = torch.sum(torch.sum((spot_y - avg_y.unsqueeze(-1)) ** 2, dim=-1), dim=-1, keepdim=True)  # [
        spotsize[:, i] = (((delta_x + delta_y) / nm / wave_num) ** (1 / 2))[:, 0]
    return spotsize, avg_y_list,avg_x_list,y_real_mw,x_real_mw


def find_best_material(rn_database, rn_model,rn_min_list,rn_max_list):

    rn_model = torch.where(rn_model > rn_max_list, rn_max_list, rn_model)
    rn_model = torch.where(rn_model < rn_min_list, rn_min_list, rn_model)


    # rn_data[材料数,3].rn_model[N,3,折射率数量] 索引13579为玻璃折射率
    wave_number = rn_model.shape[1]
    material_idx = torch.zeros((rn_model.shape[0],rn_model.shape[-1]//2)).cuda()
    for i in range(rn_model.shape[-1]):
        if  i%2!= 0:
            rn_now_glass = torch.sum((rn_model[:,:, i].unsqueeze(1) - rn_database.unsqueeze(0) ) ** 2, dim=-1)
            max_idx = torch.argmin(rn_now_glass,dim=-1,keepdim=True)  # 返回距离最近的材料的索引
            material_idx[:,i//2] = max_idx.squeeze(-1)
            # for j in range(rn_model.shape[0]):
            rn_model[:,:,i] =  torch.gather(rn_database, 0, max_idx.expand(-1, wave_number))
            # rn_model[:,:,i] = rn_database_expand[max_idx.unsqueeze(1)]
    return rn_model,material_idx



# 主函数一：光线追迹：给定上一层的光线，当前层相对上一层坐标的位置，进行坐标变换、求交点、求折射。
zero_3_dim = torch.tensor([[[0.]]]).cuda()


# 主函数二：计算近轴efl和bfl
def cal_para_focus_by_newton_multi_sys(c, th, rn, k, omit, power_para ):  # 一个问题，对于多波长，后焦距如何选择
    # rn[3,~]
    x_out = zero_3_dim.clone()
    y_plus = 0.0002
    y_out = zero_3_dim.clone() + y_plus
    z_out = zero_3_dim.clone()
    l = zero_3_dim.clone()
    m = zero_3_dim.clone()
    n = zero_3_dim.clone() + 1
    for i in range(c.shape[1]):
        x_out, y_out, z_out, l, m, n, op, a, b, d = ray_trace_newton(x_out, y_out, z_out, l, m, n, 0, 0,
                                                                     th[:, i:i + 1].unsqueeze(-1), zero_3_dim, zero_3_dim, zero_3_dim, 0,
                                                                     c[:, i:i + 1].unsqueeze(-1), rn[:, :, i:i + 1],
                                                                     rn[:, :, i + 1:i + 2], 1., 0.)
        # x_out, y_out, z_out, l, m, n, op = ray_trace_newton(x_out, y_out, z_out, l, m, n, zero_gpu, zero_gpu, th[i], zero_gpu, zero_gpu, zero_gpu, 1, c[i],rn[i], rn[i + 1])

    nu_angle = torch.arccos(n)
    EFL = (y_plus / efl_scale / nu_angle)
    BFL = y_out / nu_angle
    return EFL, BFL



# 主函数四：给定光学系统参数和光阑位置，返回出瞳位置到第一个面的距离，作为第一个距离的整个光学系统的距离
# def find_entrance(c, th, rn, stop_idx, k, omit, power_para):
#     # 第一步，寻找光阑前面光学系统的像方主点
#     batch = c.shape[0]
#     th_temp = torch.cat((torch.zeros((batch, 1)).cuda(), th), dim=1)
#     z_H = cal_H_of_system(c[:, :stop_idx - 1], th_temp, rn[0:1,:], k, omit, power_para)
#     rn_temp = torch.flip(rn[0:1,:stop_idx], dims=[0])
#     x_img, y_img, z_img = cal_img_point_of_a_obj(0, 0.02 / efl_scale, 0, -torch.flip(c[:, :stop_idx - 1], dims=[1]),
#                                                  torch.flip(th[:, : stop_idx - 1], dims=[1]), rn_temp,
#                                                  k, z_H[0][0], omit, power_para)
#     th_for_ray_trace = torch.cat(
#         (z_img, th[:, :stop_idx - 2], (th[:, stop_idx - 2] + th[:, stop_idx - 1]).unsqueeze(1), th[:, stop_idx:]),
#         dim=1)
#     return th_for_ray_trace



def imgd_beta(obj, cv, rn, th):
    # l' = n'/( (n'-n)/r - n/l    ) : l是正数

    l_i_last = -obj

    beta = 1
    cv = cv.unsqueeze(1)
    th = th.unsqueeze(1)
    for i in range(cv.shape[-1]):
        l = th[:, :, i:i + 1] - l_i_last
        l_i_last = rn[:, :, i + 1:i + 2] / (
                    ((rn[:, :, i + 1:i + 2]) - rn[:, :, i:i + 1]) * cv[:, :, i:i + 1] - rn[:, :, i:i + 1] / l)
        beta = beta * (-l_i_last * rn[:, :, i:i + 1] / rn[:, :, i + 1:i + 2] / l)
    return l_i_last, beta


def find_entrance(c, th, rn, stop_idx,return_beta=False):
    # 第一步，确定前面有几个面，然后把曲率倒序翻转，厚度倒序，找到光阑距离前面的一个面的距离作为初始物距。
    c_front = -torch.flip(c[:, :stop_idx - 1], dims=[1])
    th_front = torch.flip(th[:, : stop_idx - 1], dims=[1])
    rn_front = torch.flip(rn[:, 1:2, :stop_idx], dims=[-1])
    dis_entrance, beta = imgd_beta(0, c_front, rn_front, th_front)
    th_for_ray_trace = torch.cat(
        (dis_entrance[:, 0: 1, 0], th[:, :stop_idx - 2],
         (th[:, stop_idx - 2] + th[:, stop_idx - 1]).unsqueeze(1),
         th[:, stop_idx:]),
        dim=1)
    if return_beta==False:
        return th_for_ray_trace
    else:
        return th_for_ray_trace,beta

def find_entrance_with_stop_float(c, th, rn, stop_idx,return_beta=False):
    # Confirm which surfaces stop is locate
    # First convert th to global coordinates
    th_new = torch.zeros(th.shape).cuda()
    for i in range(th.shape[-1]):
        th_new[:,i] = torch.sum(th[:,:i+1],dim=-1)
    sruf_before_stop_mask = (th_new<0)

    c_new = c * sruf_before_stop_mask
    # 第一步，确定前面有几个面，然后把曲率倒序翻转，厚度倒序，找到光阑距离前面的一个面的距离作为初始物距。
    c_front = -torch.flip(c_new, dims=[1])
    th_front = torch.flip(th[:,1:], dims=[1])
    th_front = torch.cat((-th_new[:,-1:None],th_front),dim=-1)

    rn_new= rn[:,1,:th.shape[-1]+1].repeat(th.shape[0],1)
    rn_new[:,:th.shape[-1]][~sruf_before_stop_mask] = 1.
    rn_new  = rn_new.unsqueeze(1)
    rn_front = torch.flip(rn_new[:, :, :], dims=[-1])
    dis_entrance, beta = imgd_beta(0, c_front, rn_front, th_front)
    th_for_ray_trace = torch.cat(
        (dis_entrance[:, 0: 1, 0], th[:,1:]),
        dim=1)
    if return_beta==False:
        return th_for_ray_trace
    else:
        return th_for_ray_trace,beta


def find_exit(c, th, rn, stop_idx):
    c_back = c[:, stop_idx - 1:]
    th_back = th[:, stop_idx - 1:]
    rn_back = rn[:, :, stop_idx - 1:]
    dis_entrance, beta = imgd_beta(0, c_back, rn_back, th_back)
    return dis_entrance[:, :, 0]

def ray_aming_stop_init_ray(stop_size, th, sample_delta,sample_num):
        # 此时th，cv均已经倒转。th第一个厚度，一定是空气，因为光阑一定在空气里面。stop_size维度[:,1],需要考虑多系统并行的情况
        # 使用th第二个厚度（一定是玻璃）和cv前两个值，这三个数组成第一个透镜数据，可以计算透镜允许光线穿过且没有面型交叉的最高点。
        # ↑这一思路可能并不好，需要转换思路，假设光阑前的系统一定是正光焦度，那么光阑处光线角度一定小于视场角，因此以最大视场角为参考
        # 知道了最大光线角度，光阑的上下左右坐标，就可以求出第一个透镜表面的栅格坐标
        # 正向的采样栅格（参考），从上到下x从正到负，从左到右为y从正到负，那么stop处的循序也应该是这样
        # stop_size[N,1,1]
        N = stop_size.shape[0]

        xgrid = torch.linspace(1, -1, sample_delta)
        ygrid = torch.linspace(1, -1, sample_delta)
        # 采样点栅格，从上到下x从正到负，从左到右为y从正到负
        mesh_xy = torch.meshgrid(xgrid, ygrid)
        xgrid = mesh_xy[0].reshape(1, -1).cuda()
        ygrid = mesh_xy[1].reshape(1, -1).cuda()

        radius = (xgrid ** 2 + ygrid ** 2)
        mask = radius <= (1)
        xgrid = xgrid[mask].unsqueeze(0).unsqueeze(-1)
        ygrid = ygrid[mask].unsqueeze(0).unsqueeze(-1)
        stop_cord_x = stop_size * xgrid
        stop_cord_y = stop_size * ygrid
        stop_sample_num = mask.sum()
        lens_grid_max = stop_size + th[:, 0:1, None] * math.tan(30 / 180 * math.pi)


        xgrid = torch.linspace(1, -1, sample_num).cuda()
        ygrid = torch.linspace(1, -1, sample_num).cuda()
        # 采样点栅格，从上到下x从正到负，从左到右为y从正到负
        mesh_xy = torch.meshgrid(xgrid, ygrid)
        xgrid = mesh_xy[0].reshape(1, -1).cuda()
        ygrid = mesh_xy[1].reshape(1, -1).cuda()
        radius = (xgrid ** 2 + ygrid ** 2)
        mask = radius <= (1)
        xgrid = xgrid[mask].reshape(1, 1, -1) * lens_grid_max
        ygrid = ygrid[mask].reshape(1, 1, -1) * lens_grid_max
        zgrid = torch.zeros(xgrid.shape).cuda()
        length = ((stop_cord_x - xgrid) ** 2 + (stop_cord_y - ygrid) ** 2 + (th[:, 0:1, None]) ** 2) ** 0.5
        Lgrid = (xgrid - stop_cord_x) / length
        Mgrid = (ygrid - stop_cord_y) / length
        Ngrid = (zgrid + th[:, 0:1, None]) / length

        xgrid = xgrid.repeat_interleave(stop_sample_num, dim=1).view(N, 1, -1);
        ygrid = ygrid.repeat_interleave(stop_sample_num, dim=1).view(N, 1, -1);
        zgrid = zgrid.repeat_interleave(stop_sample_num, dim=1).view(N, 1, -1);
        Lgrid = Lgrid.view(N, 1, -1);
        Mgrid = Mgrid.view(N, 1, -1);
        Ngrid = Ngrid.view(N, 1, -1);
        # stop_sample_num是光阑处采样了多少光线
        return xgrid, ygrid, zgrid, Lgrid, Mgrid, Ngrid,stop_sample_num,stop_cord_x,stop_cord_y



# 4.1 寻找光阑前面光学系统的像方主点
def cal_H_of_system(c, th, rn, k, omit, power_para):
    x = 0.
    y = 0.02 / efl_scale
    z = 0.
    l = 0.
    m = 0.
    n = 1.
    for i in range(c.shape[1]):
        x, y, z, l, m, n, opl, loss_intersect, refract_loss, loss_curve = ray_trace_newton(x, y, z, l, m, n, 0, 0,
                                                                                           th[:, i:i + 1], 0, 0,
                                                                                           0, 0, c[:, i:i + 1],
                                                                                           rn[:, i:i + 1],
                                                                                           rn[:, i + 1:i + 2], 1., 0.)
    s = (0.02 - y) / m
    z = z + n * s
    return z


# 4.2 计算给定物点的像点位置信息
# def cal_img_point_of_a_obj(x, y, z, c, th, rn, k, z_pos_of_H, omit, power_para):
#     x_para = x
#     y_para = y
#     z_para = z
#     l_para = 0
#     m_para = math.sin(0.01)
#     n_para = math.cos(0.01)
#     for i in range(c.shape[1]):
#         x_para, y_para, z_para, l_para, m_para, n_para, op, loss_intersect, refract_loss, loss_curve = ray_trace_newton(
#             x_para, y_para, z_para, l_para, m_para,
#             n_para, 0, 0, th[:, i:i+1], 0, 0, 0, 0, c[:, i:i+1],
#             rn[0:1,i:i+1], rn[0:1,i + 1:i+2], 1., 0.)
#     z_H = th[:, 1].unsqueeze(1) - z_pos_of_H
#     s_H = (y ** 2 + (z - z_H) ** 2) ** 0.5
#     l_H = 0
#     m_H = -y / s_H
#     n_H = (z_H - z) / s_H
#     for i in range(c.shape[1]):
#         x, y, z, l_H, m_H, n_H, op, loss_intersect, refract_loss, loss_curve = ray_trace_newton(x, y, z, l_H, m_H, n_H,
#                                                                                                 0, 0,
#                                                                                                 th[:, i:i+1],
#                                                                                                 0, 0, 0, 0,
#                                                                                                 c[:,i:i+1],
#                                                                                                 rn[0:1,i:i+1],
#                                                                                                 rn[0:1,i+1:i+2], 1., 0.)
#     s = ((z_para - z) / n_H - (y_para - y) / m_H) / (m_para / m_H - n_para / n_H)
#     x_inter = x_para
#     y_inter = y_para + m_para * s
#     z_inter = z_para + n_para * s
#     return x_inter, y_inter, z_inter
def cal_img_point_of_a_obj(x, y, z, c, th, rn):
    # 使用z轴上的小角度边缘光线进行追迹
    x_para = x
    y_para = y
    z_para = z
    l_para = 0
    m_para = math.sin(0.01)
    n_para = math.cos(0.01)
    th_obj = th.clone()
    th_obj[:, 0] = th_obj[:, 0] + z
    # 光线追迹得到最后一个面上的点的y值和方向余弦
    for i in range(c.shape[1]):
        x_para, y_para, z_para, l_para, m_para, n_para, op, loss_intersect, refract_loss, loss_curve = ray_trace_newton(
            x_para, y_para, 0., l_para, m_para,
            n_para, 0, 0, th_obj[:, i:i + 1], 0, 0, 0, 0, c[:, i:i + 1],
            rn[0:1, i:i + 1], rn[0:1, i + 1:i + 2], 1., 0.)
    # 知道最后一面到的光线高度和方向余弦m，可以知道像面距离最后一个面多远
    img_d = y_para * n_para / m_para

    return img_d

def from_disp_formula2rn_vd_sellmeier(wave_length,k,L):
    # 从色散公式的参数计算折射率,k L[:,4],wave[1,3]
    rn_2_1 = 0
    for i in range(k.shape[-1]):
        rn_2_1 = k[:,i:i+1] * wave_length**2 /( wave_length**2 - L[:,i:i+1] )+ rn_2_1
    rn = torch.sqrt(rn_2_1+1)
    rn_2_1 = 0
    wave_length = wave_FdC
    for i in range(k.shape[-1]):
        rn_2_1 = k[:,i:i+1] * wave_length**2 /( wave_length**2 - L[:,i:i+1] )+ rn_2_1
    rn_FdC = torch.sqrt(rn_2_1 + 1)
    vd = (rn_FdC[:,1:2] - 1) / (rn_FdC[:,0:1] - rn_FdC[:,2:3])
    return rn,vd

def from_disp_formula2rn_vd_schott(wave_length,k,L):
    # 从色散公式的参数计算折射率,k L[:,4],wave[1,3]
    rn = k[:,0:1] + L[:,0:1]*(wave_length**2) + K[:,1:2]*(wave_length**-2) + L[:,1:2]*(wave_length**-4)  + \
         K[:,2:3]*(wave_length**-6)  + L[:,2:3]*(wave_length**-8)
    rn = torch.sqrt(rn)
    wave_length = wave_FdC
    rn_FdC = k[:,0:1] + L[:,0:1]*(wave_length**2) + K[:,1:2]*(wave_length**-2) + L[:,1:2]*(wave_length**-4)  + \
         K[:,2:3]*(wave_length**-6)  + L[:,2:3]*(wave_length**-8)
    rn_FdC = torch.sqrt(rn_FdC)
    vd = (rn_FdC[:,1:2] - 1) / (rn_FdC[:,0:1] - rn_FdC[:,2:3])
    return rn,vd


# 联合优化函数
def entrance_gird2opd_init(xgrid,ygrid,Ngrid,x0_object,y0_object,field_x,field_y,obj=-10e7):
    # 此时xgrid的形状不对劲，是{N，fieldnum,sample]
    field_num = len(field_y)
    N,_,_ = xgrid.shape
    xgrid = xgrid.reshape(N,field_num,-1)
    ygrid = ygrid.reshape(N, field_num, -1)
    Ngrid = Ngrid.reshape(N, field_num, -1)
    if field_x != None:
        field_y = torch.tensor(field_y).cuda()
        field_x = torch.tensor(field_x).cuda()
        field = torch.meshgrid(field_x, field_y)
        field_x = field[0].contiguous().view(-1, 1)
        field_y = field[1].contiguous().view(-1, 1)
    else:
        field_y = torch.tensor(field_y).unsqueeze(1).cuda()
        field_x = torch.zeros(field_y.shape).cuda()
    x0_object = obj * torch.tan(field_x / 180 * math.pi)  # 0.
    z0_object = obj
    y0_object = obj * torch.tan(field_y / 180 * math.pi)
    x0_object = x0_object.unsqueeze(0)
    y0_object = y0_object.unsqueeze(0)
    theta_each_field = torch.arctan(torch.sqrt(x0_object ** 2 + y0_object ** 2) / obj)
    theta_each_field = torch.where((x0_object < 0), -theta_each_field, theta_each_field)
    # 计算每个视场点的零光程差线的垂线

    k = -y0_object / x0_object
    d = (k * ygrid - xgrid) / torch.sqrt(k ** 2 + 1)
    op_init = -torch.sin(theta_each_field) * d
    op_init_y = torch.sin(field_y / 180 * math.pi) * ygrid
    op_init_x = torch.sin(field_x / 180 * math.pi) * xgrid
    y0_object = y0_object.repeat(1,1, Ngrid.shape[-1])
    x0_object = x0_object.repeat(1,1, Ngrid.shape[-1])

    # if torch.abs(x0_object).sum()!=0:
    op_init = torch.where(x0_object == 0., op_init_y, op_init)
    op_init = torch.where(y0_object == 0., op_init_x, op_init)
    # 即使光线瞄准了，中心视场的初始光程也应该为0
    center_idx = op_init.shape[-1]//2
    op_init = op_init - op_init[:,:,center_idx:center_idx+1]
    op_init = op_init.reshape(N,1,-1)
    return op_init

def read_galss_excel(file_path):
    data = pd.read_excel(file_path)
    data = data.values

    # glass_name = data[:,0:1]
    # data_PKL = data[:, 1:].astype(np.float64)
    # P = torch.from_numpy(data_PKL[:, :1])
    # K = torch.from_numpy(data_PKL[:, 1:4])
    # L = torch.from_numpy(data_PKL[:, 4:])
    # return glass_name,P,K,L
    return data

# 绘制镜头的函数

def plot_spot(x_all, y_all, target_efl,half_apture_max,field,field_num, file_name,wave_num=3):
    nm = int(x_all.shape[-1] / field_num)
    for num_sys in range(x_all.shape[0]):
        # try:
        x = x_all[num_sys]
        y = y_all[num_sys]

        spotsize = torch.zeros(1, field_num).cuda()
        avg_y_list = spotsize.clone()  # torch.zeros(batch, field_num).cuda()
        spot_x_list = torch.zeros(field_num, nm*wave_num).cuda()
        spot_y_list = torch.zeros(field_num, nm*wave_num).cuda()
        for i in range(field_num):
            spot_x = x[ :, nm * i:nm * (i + 1)].reshape(1,-1)
            spot_y = y[:, nm * i:nm * (i + 1)].reshape(1, -1)
            spot_x_list[i] = spot_x
            spot_y_list[i] = spot_y

            sum_x = torch.sum(spot_x)
            sum_y = torch.sum(spot_y)
            avg_x = sum_x / nm / wave_num  # [batch,1]
            avg_y = sum_y / nm / wave_num  # avg_y_list[:,i] = avg_y[:,0]
            avg_y_list[0, i] = avg_y
            delta_x = torch.sum((spot_x - avg_x) ** 2)  # [batch,1]
            delta_y = torch.sum((spot_y - avg_y) ** 2)  # [batch,1]
            spotsize[0, i] = (((delta_x + delta_y) / nm / wave_num) ** (1 / 2))
        print(spotsize)
        if field_num==6:
            col_num = 3
            row_num = 2
        else:
            col_num = 2
            row_num = 2
        fig, axs = plt.subplots(row_num, col_num)
        spotsize = spotsize.detach().cpu().numpy()
        avg_y_list = avg_y_list.detach().cpu().numpy()
        spot_x_list = spot_x_list.detach().cpu().numpy()
        spot_y_list = spot_y_list.detach().cpu().numpy()
        scale = 3
        size = spotsize[0][-1]
        # field = [0.00,5.00,10.00,15.00]
        theta = np.linspace(0, 2 * np.pi, 100)
        limit_r = 1.22*0.000588*target_efl/2/half_apture_max
        field_np = np.round(np.array(field),2)
        for i in range(row_num):
            for j in range(col_num):
                axs[i, j].scatter(spot_x_list[col_num * i + j], spot_y_list[col_num * i + j], s=12, color='#ff5b00')
                axs[i, j].set_title(str(field_np[col_num * i + j]) + '°', fontsize=24, fontweight='bold')
                # axs[i,j].set_xlim(0 - scale*spotsize[0][0], 0 + scale*spotsize[0][0])
                # axs[i,j].set_ylim(avg_y_list[0][0] - scale*spotsize[0][0], avg_y_list[0][0] + scale*spotsize[0][0])
                axs[i, j].set_xlim(0 - scale * size, 0 + scale * size)
                axs[i, j].set_ylim(avg_y_list[0][col_num * i + j] - scale * size,
                                   avg_y_list[0][col_num * i + j] + scale * size)

                axs[i, j].set_xlabel(str(np.round(spotsize[0][col_num * i + j] * 1000, 2)) + 'um', fontsize=24, fontweight='bold')
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                # axs[i, j].set_ylabel('image:'+str(np.round(avg_y_list[0][col_num*i+j],3))+'mm',fontsize=8)
                x_cir = np.cos(theta) * limit_r
                y_cir = np.sin(theta) * limit_r

                axs[i, j].plot(x_cir, y_cir + avg_y_list[0][col_num * i + j], color='blue', linewidth=2)
                axs[i, j].set_aspect('equal', adjustable='box')

            plt.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.savefig('../data/plot_lens_fig/'+file_name+'/spot_'+str(num_sys)+'.png', dpi=400,bbox_inches='tight')
        # except:
        #     pass

def plot_lens(y_max_list_all,th_for_plot,half_apture_max,cv,k,power,y_zero,z_zero,file_name,sample_delta):
    N = y_max_list_all.shape[0]
    for i in range(N):
        y_max_list = y_max_list_all[i] * 1.05
        fig, ax = plt.subplots(figsize=(5, 8))

        # 需要判断光阑在哪里，然后画出光阑位置
        plt.plot([-th_for_plot[i, 0], -th_for_plot[i, 0]], [half_apture_max, half_apture_max*1.5], color='black', linewidth=1)
        plt.plot([-th_for_plot[i, 0], -th_for_plot[i, 0]], [-half_apture_max, -half_apture_max *1.5], color='black', linewidth=1)

        plot_lay_out(th_for_plot[i], cv[i], y_max_list,
                     k[i], power[i], y_zero[i], z_zero[i], name=str(i),file_name=file_name,sample_delta=sample_delta)
        plt.close(fig)
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF00FF', '#00FF00', '#FFA500']
def plot_ray(z_pos_list,y_list,z_list,sample_delta):
    j=-1
    for i in range(y_list[0].shape[0]):
        if (i)%sample_delta==0:
            j = j+1
        color = color_list[j]
        plt.plot([ z_list[0][i] + z_pos_list[0],  z_list[1][i] + z_pos_list[1] ], [y_list[0][i],y_list[1][i]],color=color,linewidth=linewidth)

linewidth =1
def plot_lay_out(th_list,c_list,y_max_list,k_list,power_para_list,y_zero, z_zero,name,file_name,sample_delta):
    # 画光阑
    surf_num = c_list.shape[0]
    if th_list[0]>0:
        plot_ray([-th_list[0],0], y_zero[0:0 + 2, :], z_zero[0:0 + 2, :],sample_delta)
    else:
        pass

    y_zero = y_zero[1:,:]
    z_zero = z_zero[1:,:]
    th_list[0]=0.
    z_pos_list = np.zeros(th_list.shape)


    for i in range(th_list.shape[0]):
        z_pos_list[i] = np.sum(th_list[:i+1])
    for i in range(surf_num):
        if i%2==0:
            pass
        else:
            plot_one_len(z_pos_list[i-1:i+1],c_list[i-1:i+1],y_max_list[i-1:i+1],k_list[i-1:i+1],power_para_list[i-1:i+1] )


    plot_one_surf(z_pos_list[-1], 0, 1, y_max_list[-1], np.array([[0]]))

    for i in range(surf_num):

        plot_ray(z_pos_list[i:i+2], y_zero[i:i+2, :], z_zero[i:i+2, :],sample_delta)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../data/plot_lens_fig/'+file_name+'/' + str(name) + '.png', dpi=400, bbox_inches='tight')    # plt.show()


def plot_one_len(z_pos_list,c_list,y_max_list,k_list,power_para_list):

    z_1_min ,z_1_max = plot_one_surf(z_pos_list[0], c_list[0], k_list[0], y_max_list[0], power_para_list[0:1])# 返回镜头上下两端的z值
    z_2_min ,z_2_max = plot_one_surf(z_pos_list[1], c_list[1], k_list[1], y_max_list[1], power_para_list[1:2])
    y_max = max(y_max_list[0],y_max_list[1])
    if y_max_list[0]>y_max_list[1]:
        plt.plot([z_2_max, z_2_max], [y_max_list[0], y_max_list[1]], color='black',)
        plt.plot([z_2_max, z_2_max], [-y_max_list[1], -y_max_list[0]], color='black')
    else:
        plt.plot([z_1_max, z_1_max], [y_max_list[0], y_max_list[1]], color='black')
        plt.plot([z_1_max, z_1_max], [-y_max_list[1], -y_max_list[0]], color='black')

    plt.plot(  [z_1_max,z_2_max] , [y_max,y_max],color='black' )
    plt.plot([z_1_min, z_2_min], [-y_max, -y_max], color='black')

def plot_one_surf(z_pos,c,k,y_max,power_para=None):
    y = np.linspace(-y_max, y_max, 100)
    y2 = y**2

    power_num = power_para.shape[1]
    level = np.zeros((1, power_num))
    for j in range(power_num):
        i = j + 2
        level[0, j] = i
    y2_high_level = np.expand_dims(y2, axis=1)
    y2_high_level = np.tile(y2_high_level, (1, power_num))
    high_value = power_para * (y2_high_level ** level)
    high_value = np.sum(high_value, axis=1)

    z = (c * y2) / (1 + (1 - k * c ** 2 * y2) ** 0.5) + high_value + z_pos
    plt.plot(z, y,color = 'black')
    return z[0],z[-1]








wave_FdC = torch.tensor([[0.486,0.588,0.656]]).cuda()
wave_sys_for_rn = torch.tensor([[0.486,0.588,0.656]]).cuda()


rn_repeat_num = 1

# wave_sys_for_rn = torch.linspace(0.486,0.656,wave_number).unsqueeze(0).cuda()
wave_sys_for_rn = wave_sys_for_rn.repeat(1,rn_repeat_num)
wave_sys = (wave_sys_for_rn/1000).unsqueeze(-1).unsqueeze(-1)

glass_data = read_galss_excel('../data/CDGM_C10.xlsx')
data_PKL = glass_data[:,2:].astype(np.float64)
formula_type = glass_data[:,1]
mask_sellm = (formula_type==' Sellmeier1')
mask_schot = (formula_type=='     Schott')

# data_PKL = data_PKL
P = torch.from_numpy(data_PKL[:,3:4]).cuda()
K = torch.from_numpy(data_PKL[:,4:7]).cuda()
L = torch.from_numpy(data_PKL[:,7:]).cuda()


rn_cdgm_scot_sem,vd_0417_sem = from_disp_formula2rn_vd_sellmeier(wave_sys_for_rn,K,L)
rn_cdgm_scot_schot,vd_0417_schot = from_disp_formula2rn_vd_schott(wave_sys_for_rn,K,L)
rn_cdgm_scot_sem[mask_schot] = rn_cdgm_scot_schot[mask_schot]
rn_cdgm_scot = rn_cdgm_scot_sem
vd_0417_sem[mask_schot] = vd_0417_schot[mask_schot]
vd_0417 = vd_0417_sem

rn_database = rn_cdgm_scot.cuda()
indices = torch.sort(rn_database[:,2]).indices
rn_database  = rn_database[indices]
rn_database_num = rn_database.cpu().numpy()
vd_0417 = vd_0417[indices]
glass_name = glass_data[:,0][indices.cpu()].tolist()
glass_name = [item.strip() for item in glass_name]

# glass_name = glass_data[:,0].tolist()


if __name__ == "__main__":
    omit = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # rn = find_best_material(rn_UV,rn_test)
    print(rn_cdgm_scot)








    #
    #
    # N = 2
    # cv = 1 / torch.tensor(
    #     [[21.48138, -124.1, -19.1, 22, 328.9, -16.7], [21.48138, -124.1, -19.1, 22, 328.9, -16.7]]).cuda()
    #
    # # th = torch.tensor([[0, 2, 5.26, 1.25, 4., 2.25], [0, 2, 5.26, 1.25, 4.69, 2.25]]).cuda()
    # th = torch.tensor([[2, 5.26, 1, 0.25, 4.69, 2.25], [2, 5.26, 1.2, 0.5, 4.69, 2.25]]).cuda()
    #
    # rn = torch.tensor([[[1, 1.6204099651, 1, 1.6165918071, 1, 1.6204099651, 1, 1, 1, 1],
    #                     [1, 1.6275563488, 1, 1.6284791544, 1, 1.6275563488, 1, 1, 1, 1],
    #                     [1, 1.6172716599, 1, 1.6116452953, 1, 1.6172716599, 1, 1, 1, 1]]]).cuda()
    #
    # k = torch.zeros((N, 6), dtype=torch.float64).cuda()
    # # power_para[0][0] = 0.001
    # stop_idx = 4
    #
    # field_num = 3
    # half_apture_min = 2.4
    # field_x = [0, 0, 0]
    # field_y = [0, 5, 10]
    # field = [0, 5, 10]
    # sample_delta = 21
    # th_label = [1, 0, 3, 3, 0, 1]
    # power_para = torch.zeros((N, 6, 1), dtype=torch.float64).cuda()
    # obj = 10e6
    # # img_d = cal_img_point_of_a_obj(0, 0,obj , cv, th, rn)
    # # print('img d:', img_d)
    # # 计算光学系统点列图
    # x_init, y_init, z_init, l_init, m_init, n_init, number_of_gridpoints, sample_delta = grid_data_of_staring_point_para(
    #     half_apture_min, field_y,
    #     sample_delta=sample_delta, field_x=None, obj=-obj)
    # if stop_idx != 1:
    #     # th_for_ray_trace = find_entrance(cv, th, rn, stop_idx, k, omit, power_para)
    #     th_for_ray_trace = find_entrance(cv, th, rn, stop_idx)
    #     exit_dist = find_exit(cv, th, rn, stop_idx)
    # else:
    #     th_for_ray_trace = th
    # efl, bfl = cal_para_focus_by_newton_multi_sys(cv, th_for_ray_trace, rn, k, omit, power_para)
    # x_out, y_out, z_out, l, m, n = x_init.data, y_init.data, z_init.data, l_init.data, m_init.data, n_init.data
    # for i in range(cv.shape[1]):
    #     x_out, y_out, z_out, l, m, n, op, loss_intersect_layer, loss_refract_layer, loss_curve = \
    #         ray_trace_newton(x_out, y_out, z_out, l, m, n, 0, 0, th_for_ray_trace[:, i:i + 1].unsqueeze(-1),
    #                          0, 0, 0,
    #                          k[:, i:i + 1].unsqueeze(-1), cv[:, i:i + 1].unsqueeze(-1),
    #                          rn[:, :, i:i + 1],
    #                          rn[:, :, i + 1:i + 2], omit[i], power_para[:, i:i + 1, :],
    #                          th_label[i])
    # # x_out, y_out, z_out, l, m, n, op, loss_intersect_layer, loss_refract_layer, loss_curve = \
    # #     ray_trace_newton(x_out, y_out, z_out, l, m, n, 0, 0, -img_d.unsqueeze(1), 0, 0, 0, 0, 0.,
    # #                       rn[:, -2:-1],  rn[:, -2:-1], 1, 0)
    # x_out, y_out, z_out, l, m, n, op, loss_intersect_layer, loss_refract_layer, loss_curve = \
    #     ray_trace_newton(x_out, y_out, z_out, l, m, n, 0, 0, bfl[:, 0:1, :], 0, 0, 0, 0, 0.,
    #                      rn[:, :, -2:-1], rn[:, :, -2:-1], 1, 0)
    #
    # spotsize, avg_y_list, va = cal_mlzi_spot_size(x_out, y_out, field_num)
    # print(spotsize)
