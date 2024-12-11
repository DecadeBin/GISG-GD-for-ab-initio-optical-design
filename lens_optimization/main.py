import sys

import torch.nn
from torch import optim

# from  DFRT_fisheye_joint import *
from lens_para import *
import torch.nn as nn
import pickle
import time
import random
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)


# 光学设计算法函数 ******************************************************************************************************************
torch.set_default_dtype(torch.float64)

# 设置每类变量的学习率
def lr_set(wight_c,wight_th,wight_k,wight_power,weight_rn,lr_max,weight_grd):

    lr_c = lr_max * wight_c * weight_grd
    lr_th = lr_max * wight_th * weight_grd
    lr_k = lr_max * wight_k * weight_grd
    lr_power = lr_max * wight_power * weight_grd
    lr_rn = lr_max * weight_rn * weight_grd
    return  lr_c,lr_th,lr_k,lr_power,lr_rn
# 生成各个变量的最大最小值
def value_max_min(length_num,th_label,th_g_min,th_g_max,th_a_min,th_a_max,th_stop_min,th_stop_max,
                  k_label,k_as_min,k_as_max,power_num,surf_num,power_max_value,rn_min,rn_max):
    power_max = torch.zeros(1, 1, power_num).cuda()
    for i in range(power_num):
        power_max[0, 0, i] = power_max_value * ((2 * i + 3))
    power_max = power_max.repeat_interleave(surf_num, dim=1)
    power_max = power_max.repeat_interleave(op_args.N, dim=0)

    th_min = torch.zeros((1, length_num)).cuda()
    for i in range(length_num):
        if th_label[i] == 1:
            th_min[0][i] = th_g_min
        elif th_label[i] == 0:
            th_min[0][i] = th_a_min
        else:
            th_min[0][i] = th_stop_min
    th_max = torch.zeros((1, length_num)).cuda()
    for i in range(length_num):
        if th_label[i] == 1:
            th_max[0][i] = th_g_max
        elif th_label[i] == 0:
            th_max[0][i] = th_a_max
        else:
            th_max[0][i] = th_stop_max
    k_max = torch.zeros((1, length_num)).cuda()
    k_min = torch.zeros((1, length_num)).cuda()
    if lr_k!=0:
        for i in range(length_num):
            if k_label[i] == 1:
                k_max[0][i] = k_as_max
                k_min[0][i] = k_as_min
            else:
                k_max[0][i] = 0.
                k_min[0][i] = 0.
    rn_max_list = torch.ones((1, length_num+1)).cuda()
    rn_min_list = torch.ones((1, length_num+1)).cuda()
    for i in range(length_num):
        if i%2!=0:
            rn_max_list[0,i] =  rn_max
            rn_min_list[0, i] = rn_min


    th_min = th_min.repeat_interleave(op_args.N, dim=0)
    th_max = th_max.repeat_interleave(op_args.N, dim=0)
    if op_args.file_name=='6p':
        th_max[:,-1] = 0.3
    if lr_k!=0:
        k_max = k_max.repeat_interleave(op_args.N, dim=0)
        k_min = k_min.repeat_interleave(op_args.N, dim=0)
    else:
        k_max = 0.
        k_min = 0.
    return th_min,th_max,k_min,k_max,power_max,rn_min_list,rn_max_list
# 生成厚度标签
def generate_th_label(surf_num,stop_idx):
    th_label = np.zeros(surf_num)
    if stop_idx==1:
        th_label[0] = 3
        for i in range(surf_num-1):
            if i %2==0:
                th_label[i+1] = 1
            else:
                th_label[i+1] = 0
    else:
        if stop_idx%2==1:
            for i in range(surf_num):
                if (i<(stop_idx-2)) & (i%2==0):
                    th_label[i ] = 1
                elif (i<(stop_idx-2)) & (i%2!=0):
                    th_label[i ] = 0
                elif ( i>(stop_idx-1)) & (i%2==0):
                    th_label[i ] = 0
                elif (i>(stop_idx-1)) & (i%2!=0):
                    th_label[i ] = 1
                else:
                    th_label[i] = 3
        else:
            for i in range(surf_num):
                if (i<(stop_idx-2)) & (i%2==0):
                    th_label[i ] = 1
                elif (i<(stop_idx-2)) & (i%2!=0):
                    th_label[i ] = 0
                elif ( i>(stop_idx-1)) & (i%2==0):
                    th_label[i ] = 0
                elif (i>(stop_idx-1)) & (i%2!=0):
                    th_label[i ] = 1
                else:
                    th_label[i] = 3


    return th_label
# 生成速度最大最小值
def v_max_min(N,surf_num,v_c_max,length_num,v_th_max,v_rn_max,weight_grd,v_k_max):
    v_c = torch.rand((N, surf_num)) * v_c_max * 2 - v_c_max
    v_c = (1 - weight_grd) * v_c.cuda()
    v_th = torch.rand([N, length_num]) * v_th_max * 2 - v_th_max
    v_th = (1 - weight_grd) * v_th.cuda()
    v_k = torch.rand([N, surf_num], dtype=torch.float64) * v_k_max * 2 - v_k_max
    v_k = (1 -weight_grd) * v_k.cuda()
    v_rn = torch.rand([N, lens_args.wave_number,length_num+1]) * v_rn_max * 2 - v_rn_max
    v_rn = (1 -weight_grd) * v_rn.cuda()
    return v_c,v_th,v_rn,v_k
# 寻找无光线追迹错误的起始点
def find_init():

    # c_init_2 = torch.zeros(init_num * op_args.N, lens_args.surf_num)
    # th_init_2 = torch.zeros(init_num * op_args.N, lens_args.length_num)
    model_init = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, th_label,
                                           omit=lens_args.power_omit, field_num=op_args.field_num, stop_idx=lens_args.stop_idx,
                                           wave_number=lens_args.wave_number).cuda()


    # op_args.N = 1
    # model = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, len_s_diameter,
    #                                        omit=lens_args.power_omit, op_args.field_num=op_args.field_num, lens_args.stop_idx=lens_args.stop_idx).cuda()
    # model.cv.data = 1/torch.tensor([[ 2.810497748, -14.70635105, -2.762818125, -2.693013695, 2.970276092, 8.872489624, -2.442339785, -3.470315537,0,0]]).cuda()
    #
    # model.th.data = torch.tensor([[ 0, 0.95856815, 0.194808341, 1, 0.177171277, 0.31843212, 0.8, 0.595001355, 0.010004858, 0.15]]).cuda()
    # model.k.data = torch.tensor([[-2.43994189, -0.454307397, -2.699659179, 0.217803331, -3.360098506, 8.512134899, 0.424892034, -6.33650586,0,0]]).cuda()
    # model.power_para.data[0,0,-1] =100
    # spot_size, bfl, efl, y_field, loss_intersect, y_surf_height_loss, avg_y_list,a = model(x_init, y_init, z_init, l_init, m_init, n_init)
    loss = torch.ones([op_args.N, 1]) * 1e8
    loss = loss.cuda()
    init_num = 20  # 找多少倍个焦距合适的算点列图
    num1 = 20 # 几倍的数量，用来算焦距合适
    start = time.time()
    rn_number = lens_args.surf_num//2
    # model_init.decenter_z = torch.zeros([num1 * init_num * op_args.N, 1, lens_args.length_num]).cuda()
    with torch.no_grad():
        find_init_loop_is_false = 0
        try_time=0
        k_label_tensor = torch.tensor(lens_args.k_label).unsqueeze(0)
        while find_init_loop_is_false==0:
            temp1 = 0
            right = 0
            if op_args.rn_vara==True:
                rn_init = torch.ones(num1 * init_num * op_args.N, lens_args.wave_number, lens_args.surf_num + 1).cuda()
            while right == 0:
                c_init = (torch.rand((num1 * init_num * op_args.N, lens_args.surf_num),
                                     ) * lens_args.c_init_max * 2 - lens_args.c_init_max).data

                c_init = c_init.cuda()
                # c_init[:,6]  =  - torch.abs(c_init[:,6] )
                if op_args.file_name=='6p':
                    c_init[:,-2:]  = 0.
                if op_args.rn_vara == True:
                    rn_init_idx = torch.randint(0,rn_database.shape[0],(num1 * init_num * op_args.N,rn_number)).cuda()
                    for i in range(lens_args.surf_num + 1):
                        if i%2 != 0:
                            rn_init[:, :, i] = torch.gather(rn_database, 0, rn_init_idx[:,i//2:i//2+1].expand(-1, lens_args.wave_number))
                    model_init.rn.data = rn_init
                # c_init[:,0] = torch.abs(c_init[:,0])
                # c_init[:, 10] = -torch.abs(c_init[:, 10])
                # th_init = (torch.rand([num1*init_num*op_args.N, lens_args.length_num], dtype=torch.float64) * (th_max - th_min) + th_min).data
                th_init = (torch.rand((num1 * init_num * op_args.N, lens_args.length_num)) * (
                        th_max.repeat_interleave(num1 * init_num, dim=0).cpu() - th_min.repeat_interleave(num1 * init_num,
                                                                                                          dim=0).cpu())
                           + th_min.repeat_interleave(num1 * init_num, dim=0).cpu()).data

                if lr_k!=0 and op_args.k_if_init:

                    k_init = (torch.rand((num1 * init_num * op_args.N, lens_args.surf_num),
                                         ) * lens_args.k_init_max * 2 - lens_args.k_init_max).data * k_label_tensor



                # th_init[:,2]=0.
                th_init = th_init.cuda()
                model_init.cv.data = c_init
                model_init.th.data = th_init
                if lr_k != 0 and op_args.k_if_init:
                    model_init.k.data = k_init
                bfl, efl = model_init.forward_efl(x_init, y_init, z_init, l_init, m_init, n_init)
                mask = ((bfl[:, op_args.wave_idx, :] < lens_args.bfl_max) & (bfl[:, op_args.wave_idx, :] > lens_args.bfl_min) & (
                        torch.abs(efl[:, op_args.wave_idx, :] - lens_args.target_efl) < (0.3 * lens_args.target_efl))).squeeze(1)
                c_temp = c_init[mask, :]
                th_temp = th_init[mask, :]
                if lr_k!=0 and op_args.k_if_init:
                    k_temp = k_init[mask, :]
                if op_args.rn_vara == True:
                    rn_temp = rn_init[mask, :,:]
                else:
                    rn_temp = torch.tensor([0])
                if temp1 == 0:
                    c_init_2 = c_temp
                    th_init_2 = th_temp
                    rn_init_2 = rn_temp
                    if lr_k!=0 and op_args.k_if_init:
                        k_init_2 = k_temp
                    temp1 = 1
                else:
                    c_init_2 = torch.cat((c_init_2, c_temp), dim=0)
                    th_init_2 = torch.cat((th_init_2, th_temp), dim=0)
                    rn_init_2 = torch.cat((rn_init_2, rn_temp), dim=0)

                print(c_init_2.shape[0],'  : ',end='')
                # print(bfl[mask][:, 0, :])

                if c_init_2.shape[0] >= init_num * op_args.N:
                    c_init_2 = c_init_2[:init_num * op_args.N, :]
                    th_init_2 = th_init_2[:init_num * op_args.N, :]
                    if lr_k!=0 and op_args.k_if_init:
                        k_init_2 = k_init_2[:init_num * op_args.N, :]
                    right = 1
                    break
            ####################################################################################################################################################################################################################
            ####################################################################################################################################################################################################################

            model_init = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, th_label,
                                                   omit=lens_args.power_omit, field_num=op_args.field_num, stop_idx=lens_args.stop_idx,
                                                   wave_number=lens_args.wave_number).cuda()
            temp1 = 0
            for j in range(init_num):
                # print(j,':',end='')
                model_init.cv.data = c_init_2[j * op_args.N:(j + 1) * op_args.N, :].cuda()
                model_init.th.data = th_init_2[j * op_args.N:(j + 1) * op_args.N, :].cuda()
                if lr_k!=0 and op_args.k_if_init:
                    model_init.k.data = k_init_2[j * op_args.N:(j + 1) * op_args.N, :].cuda()
                if op_args.rn_vara==True:
                    model_init.rn.data = rn_init_2[j * op_args.N:(j + 1) * op_args.N, :,:].cuda()
                # forward_cal
                spot_size, bfl, efl, y_field, loss_intersect, loss_angle, color_loss, loss_curve_all,opd = model_init(
                    x_init, y_init, z_init, l_init, m_init, n_init,op_init=op_init)
                spot_size_sum = torch.sum(spot_size, dim=-1, keepdim=True)

                mask =  ((spot_size_sum < op_args.init_spot_constrain)&(loss_intersect==0)).squeeze(1)
                # if torch.sum(mask)!=0:
                #     print(torch.sum(mask).item(),';',end='')

                c_temp = model_init.cv.data[mask, :]
                th_temp = model_init.th.data[mask, :]
                if lr_k!=0 and op_args.k_if_init:
                    k_temp = model_init.k.data[mask, :]
                if op_args.rn_vara == True:
                    rn_temp = model_init.rn.data[mask, :,:]
                else:
                    rn_temp = torch.tensor([0])
                if temp1 == 0:
                    c_2 = c_temp
                    th_2 = th_temp
                    rn_model_2 = rn_temp
                    if lr_k!=0 and op_args.k_if_init:
                        k_2 = k_temp
                    temp1 = 1
                else:
                    c_2 = torch.cat((c_2, c_temp), dim=0)
                    th_2 = torch.cat((th_2, th_temp), dim=0)
                    rn_model_2 = torch.cat((rn_model_2, rn_temp), dim=0)
                    if lr_k!=0 and op_args.k_if_init:
                        k_2 = torch.cat((k_2, k_temp), dim=0)

            if try_time==0:
                c = c_2
                th =th_2
                rn_model =rn_model_2
                try_time=1
                if lr_k!=0 and op_args.k_if_init:
                    k = k_2
            else:
                c = torch.cat((c, c_2), dim=0)
                th = torch.cat((th, th_2), dim=0)
                rn_model = torch.cat((rn_model, rn_model_2), dim=0)
                if lr_k!=0 and op_args.k_if_init:
                    k = torch.cat((k, k_2), dim=0)
            print('valid num: ',c.shape[0])
            if c.shape[0] >= op_args.N:
                c = c[:  op_args.N, :]
                th = th[: op_args.N, :]
                if op_args.rn_vara==True:
                    rn_model = rn_model[: op_args.N, :]
                if lr_k!=0 and op_args.k_if_init:
                    k = k[: op_args.N, :]
                find_init_loop_is_false=1
                break

    torch.cuda.empty_cache()
    end = time.time()
    print('find init time: ', end - start)
    torch.cuda.empty_cache()
    idx_best = torch.argmin(loss, dim=0)
    c_best = c[idx_best]
    th_best = th[idx_best]
    if lr_k!=0 and op_args.k_if_init:
        k_best = k[idx_best]
        k_person_best = k

    # 确定局部最优解
    c_person_best = c
    th_person_best = th
    if op_args.rn_vara==True:
        rn_best = rn_model[idx_best]
        rn_person_best = rn_model
    else:
        rn_best = torch.tensor([0])
        rn_person_best = torch.tensor([0])
    loss_person_best = loss
    loss_best_list = torch.zeros((op_args.epoch)).cuda()
    loss_best_all = 1.e8
    loss_list = []
    loss_last_one = loss
    person_best_i = torch.zeros((op_args.N, 1)).cuda()  # 用来记录每一个点最近的一次最好的点出现在哪一代
    judge_guidence = torch.zeros((op_args.N, 1)).cuda()
    if lr_k!=0 and op_args.k_if_init==False:
        k = torch.zeros((op_args.N, lens_args.surf_num)).cuda()
        k_person_best = k
        k_best = k[idx_best]
    if lr_k==0:
        k=0
        k_person_best = 0
        k_best = 0
    if lr_power!=0:
        power_temp = torch.zeros((op_args.N,lens_args.surf_num,lens_args.power_num)).cuda()
        power_person_best = power_temp
    else:
        power_temp=0
        power_person_best=0
    print('k_best:',k_best)

    return c,th,rn_model,loss,c_best,c_person_best,th_best,th_person_best,rn_best,rn_person_best,loss_person_best,loss_best_list,loss_best_all,loss_list,\
           loss_last_one,person_best_i,judge_guidence,k,k_person_best,k_best,power_temp,power_person_best
# 优化中函数*********************************************
# 使用粒子群策略更新一步参数
def pos_one_time_iter(model,v_c,v_th,v_rn,i,loss_last_one,loss_temp,v_k=0):
    c = model.cv.data
    th = model.th.data
    if lr_k != 0:
        k = model.k.data
    else:
        k=0
    if op_args.rn_vara == True:
        rn_model = model.rn.data


    # with torch.no_grad():
    if i == 0:
        c_v_temp = w * v_c + weight_local * (c_person_best - c) + weight_global * (c_best - c)
        th_v_temp = w * v_th + weight_local * (th_person_best - th) + weight_global * (th_best - th)
        if op_args.rn_vara==True:
            rn_v_temp = w * v_rn + weight_local * (rn_person_best - rn_model) + weight_global * (rn_best - rn_model)
        else:
            rn_v_temp = 0
        if lr_k!=0:
            k_v_temp = w * v_k + weight_local * (k_person_best - k) + weight_global  * (k_best - k)


    else:
        rand1 = (torch.rand((op_args.N, 1)) * 1.3 - 0.2).cuda()
        rand2 = (torch.rand((op_args.N, 1)) * 1.3 - 0.2).cuda()
        c_v_temp_not_change = w * v_c  # + weight_local * rand1 * (c_person_best - c) + weight_global * rand2 * (c_best - c)
        th_v_temp_not_change = w * v_th  # + weight_local * rand1 * (th_person_best - th) + weight_global * rand2 * (th_best - th)
        rn_v_temp_not_change = w * v_rn
        k_v_temp_not_change = w * v_k

        # c_v_temp_change = w * v_c + weight_local * rand1 * (c_person_best - c) + weight_global * rand2 * (c_best - c)
        # th_v_temp_change = w * v_th + weight_local * rand1 * (th_person_best - th) + weight_global * rand2 * (th_best - th)
        # k_v_temp_change = w * v_k + weight_local * rand1 * (k_person_best - k) + weight_global * rand2 * (k_best - k)

        c_v_temp_change = w * v_c + weight_local * rand1 * (c_person_best - c) + weight_global * rand2 * (c_best_all - c)
        th_v_temp_change = w * v_th + weight_local * rand1 * (th_person_best - th) + weight_global * rand2 * (th_best_all - th)
        k_v_temp_change = w * v_k + weight_local * rand1 * (k_person_best - k) + weight_global * rand2 * (k_best_all - k)

        if op_args.rand_guidence==True:
            v_c = torch.rand((op_args.N, lens_args.surf_num)).cuda() * op_args.v_c_max * 2 - op_args.v_c_max
            v_th = torch.rand([op_args.N, lens_args.length_num]).cuda() * op_args.v_th_max * 2 - op_args.v_th_max
            v_k = torch.rand([op_args.N, lens_args.surf_num], dtype=torch.float64).cuda() * op_args.v_k_max * 2 - op_args.v_k_max
            rand1 = (torch.rand((op_args.N, 1)) * 1.3 - 0.2).cuda()
            rand2 = (torch.rand((op_args.N, 1)) * 1.3 - 0.2).cuda()
            c_v_temp_change = w * v_c + weight_global * v_c  # rand2 * (c_best - c)
            th_v_temp_change = w * v_th  + weight_global * v_th  # rand2 * (th_best - th)
            k_v_temp_change = w * v_k  + weight_global * v_k  # rand2 * (k_best - k)

        if op_args.rn_vara==True:
            rn_v_temp_change = w * v_rn + weight_local * rand1.unsqueeze(-1) * (rn_person_best - rn_model) + weight_global * rand2.unsqueeze(-1) * (rn_best_all - rn_model)

        if op_args.method_name=='GISG_GD':
            c_v_temp = c_v_temp_change
            th_v_temp = th_v_temp_change
            if lr_k != 0:
                k_v_temp =k_v_temp_change#  torch.where(loss_temp <= loss_last_one, k_v_temp_not_change, k_v_temp_change)
            if op_args.rn_vara==True:
                rn_v_temp = rn_v_temp_change
            else:
                rn_v_temp = 0
        else:
            c_v_temp = torch.where(loss_temp <= loss_last_one, c_v_temp_not_change, c_v_temp_change)
            th_v_temp = torch.where(loss_temp <= loss_last_one, th_v_temp_not_change, th_v_temp_change)
            if lr_k != 0:
                k_v_temp = torch.where(loss_temp <= loss_last_one, k_v_temp_not_change, k_v_temp_change)
            if op_args.rn_vara==True:
                rn_v_temp = torch.where(loss_temp.unsqueeze(-1) <= loss_last_one.unsqueeze(-1) , rn_v_temp_not_change, rn_v_temp_change)
            else:
                rn_v_temp = 0
        loss_last_one = loss_temp.data

    # 计算参数更新的步长

    # 计算参数更新的步长
    c_v_temp = torch.where(c_v_temp > op_args.v_c_max, op_args.v_c_max, c_v_temp)
    c_v_temp = torch.where(c_v_temp < -op_args.v_c_max, -op_args.v_c_max, c_v_temp)
    th_v_temp = torch.where(th_v_temp > op_args.v_th_max, op_args.v_th_max, th_v_temp)
    th_v_temp = torch.where(th_v_temp < -op_args.v_th_max, -op_args.v_th_max, th_v_temp)
    if lr_k!=0:
        k_v_temp = torch.where(k_v_temp > op_args.v_k_max, op_args.v_k_max, k_v_temp)
        k_v_temp = torch.where(k_v_temp < -op_args.v_k_max, -op_args.v_k_max, k_v_temp)
    if op_args.rn_vara==True:
        rn_v_temp = torch.where(rn_v_temp > op_args.v_rn_max, op_args.v_rn_max, rn_v_temp)
        rn_v_temp = torch.where(rn_v_temp < -op_args.v_rn_max, -op_args.v_rn_max, rn_v_temp)
        rn_temp = rn_model + op_args.weight_guidence * rn_v_temp * judge_guidence.unsqueeze(-1)
        v_rn = rn_v_temp
    else:
        rn_temp = 0
        v_rn = 0
    # 参数更新
    if op_args.method_name=='GISG_GD' or op_args.method_name=='PSO_GISG_GD': # only method_name satisfy these two, use selective guidance
        c_temp = c + op_args.weight_guidence * c_v_temp * judge_guidence
        th_temp = th + op_args.weight_guidence * th_v_temp * judge_guidence
        if lr_k!=0:
            k_temp = k + op_args.weight_guidence * k_v_temp   * judge_guidence
            k_temp = torch.where(k_temp > k_max, k_max, k_temp)
            k_temp = torch.where(k_temp < k_min, k_min, k_temp)
        else:
            k_temp=0
            v_k = 0
    else:
        c_temp = c + op_args.weight_guidence * c_v_temp # * judge_guidence
        th_temp = th + op_args.weight_guidence * th_v_temp #  * judge_guidence
        if lr_k != 0:
            k_temp = k + op_args.weight_guidence * k_v_temp  # * judge_guidence
            k_temp = torch.where(k_temp > k_max, k_max, k_temp)
            k_temp = torch.where(k_temp < k_min, k_min, k_temp)
        else:
            k_temp=0
            v_k = 0
    # 赋值给v，下一次更新的基本速度
    v_c = c_v_temp
    v_th = th_v_temp
    if lr_k!=0:
        v_k = k_v_temp
    else:
        v_k = 0.

    if op_args.file_name == '6p':
        c_temp[:, -2:] = 0.

    model.cv.data = c_temp
    model.th.data = th_temp
    if lr_k != 0:
        model.k.data = k_temp
    return model,v_c,v_th,rn_temp,v_rn,loss_last_one,v_k
# 计算光学系统损失
def cal_loss(model):
    c_board_loss_1 = torch.where(model.cv > lens_args.cmax, model.cv - lens_args.cmax, 0.)
    c_board_loss_2 = torch.where(model.cv < -lens_args.cmax, -lens_args.cmax - model.cv, 0.)
    c_board_loss = torch.sum((c_board_loss_1 + c_board_loss_2), dim=-1, keepdim=True)
    th_board_loss_1 = torch.where(model.th > th_max, model.th - th_max, 0.)
    th_board_loss_2 = torch.where(model.th < th_min, th_min - model.th, 0.)
    th_board_loss = torch.sum((th_board_loss_1 + th_board_loss_2)[:, :13], dim=-1, keepdim=True)

    # try:
    if op_args.weight_grd!=0:
        # forward_cal
        spot_size, bfl, efl, y_field, loss_intersect, loss_angle, color_loss, loss_curve_all,opd = model(
            x_init, y_init, z_init, l_init, m_init, n_init,op_init=op_init)
    else:
        with torch.no_grad():
            spot_size, bfl, efl, y_field, loss_intersect, loss_angle, color_loss, loss_curve_all,opd = model(
                x_init, y_init, z_init, l_init, m_init, n_init)

    if weight_arg.weight_opd!=0:
        opd_loss = torch.sum(torch.sum(torch.max(opd,dim=-1).values - torch.min(opd,dim=-1).values,dim=-1),dim=-1,keepdim=True)
    else:
        opd_loss = 0
    dist = (y_field_ideal - y_field) / (y_field_ideal + 1e-8)
    dist_loss = torch.max(dist, dim=-1, keepdim=True).values - torch.min(dist, dim=-1, keepdim=True).values
    dist_loss = torch.where(dist_loss > lens_args.dist_max, dist_loss - lens_args.dist_max, 0.)
    bfl_loss = bfl_cal(bfl, lens_args.bfl_max, lens_args.bfl_min, weight_arg.weight_bfl)  # torch.where((bfl > lens_args.bfl_max)|(bfl<lens_args.bfl_min),,0.)
    th_for_totr = (torch.where(model.th[:, 0] > 0, model.th[:, 0], 0.)).unsqueeze(1)
    TOTR = (torch.sum(model.th[:, 1:], dim=-1, keepdim=True) + bfl[:, op_args.wave_idx:op_args.wave_idx+1, 0] + th_for_totr)
    TOTR_loss = torch.where((TOTR > lens_args.TOTR_max), TOTR - lens_args.TOTR_max, 0.)
    # c_std_loss_1 = torch.std(torch.abs(model.cv[:,:-2]), dim=1,keepdim=True)
    f = cal_f(model_cv=model.cv[:, :-2], model_th=model.th[:, :-2], model_rn=model.rn)
    if weight_arg.weight_c_std!=0:
        c_std_loss = torch.std(torch.abs(f), dim=1, keepdim=True)
    else:
        c_std_loss = 0.
    if weight_arg.weight_color!=0:
        color_loss = torch.where(color_loss > lens_args.color_max, color_loss - lens_args.color_max, 0.)
    else:
        color_loss = 0.


    # spot_std_loss = torch.std(spot_size,dim=-1,keepdim=True)
    spot_std_loss = torch.max(spot_size,dim=-1,keepdim=True).values - torch.min(spot_size,dim=-1,keepdim=True).values
    # 需要有一个使得点列图尽量均衡的指标

    # spot，焦距，有交点，全内反射，像高，总长，玻璃面交点高度，bfl，色差
    with torch.no_grad():
        loss_temp = (torch.sum((weight_spot_size * spot_size), dim=-1, keepdim=True) + weight_arg.weight_efl * torch.abs(
            efl[:, op_args.wave_idx:op_args.wave_idx+1, 0] - lens_args.target_efl) + bfl_loss * weight_arg.weight_bfl
                     + weight_arg.weight_intersect * loss_intersect
                     + weight_arg.weight_dist * dist_loss
                     + weight_arg.weight_TOTR * TOTR_loss
                     + weight_arg.weight_angle * loss_angle
                     + weight_arg.weight_curve * loss_curve_all
                     + weight_arg.weight_c_std * c_std_loss
                     + weight_arg.weight_color * color_loss
                     + weight_arg.weight_c_board * c_board_loss
                     + weight_arg.weight_th_board * th_board_loss
                     + weight_arg.weight_spot_std * spot_std_loss
                     + weight_arg.weight_opd * opd_loss
                     ).data
    # 小视场，点列图权重恒定，大视场波动
    err_num = torch.sum(torch.isnan(loss_temp))
    # print('err_num: ', err_num)
    spot_size_for_weight = (spot_size.data) ** 1.2
    weight_spot_size_b = spot_size_for_weight / torch.sum(spot_size_for_weight, dim=-1,
                                                          keepdim=True) * weight_arg.weight_spot
    loss_temp = torch.where(torch.isnan(loss_temp), 1.e3, loss_temp)
    spot_size = torch.where(torch.isnan(spot_size), 10., spot_size)
    if op_args.weight_grd!=0:

        # loss_back = loss_temp.sum().backward()

        loss_back = (weight_spot_size_b * spot_size).sum() + (
                weight_arg.weight_efl * torch.abs(efl[:, op_args.wave_idx:op_args.wave_idx+1, 0] - lens_args.target_efl) + bfl_loss * weight_arg.weight_bfl).sum() + \
                    (weight_arg.weight_intersect * loss_intersect).sum() \
                    + (weight_arg.weight_dist * dist_loss).sum() \
                    + (weight_arg.weight_TOTR * TOTR_loss + weight_arg.weight_c_std * c_std_loss + weight_arg.weight_color * color_loss+ weight_arg.weight_spot_std * spot_std_loss + + weight_arg.weight_opd * opd_loss).sum() \
                    + (weight_arg.weight_angle * loss_angle).sum() \
                    + (weight_arg.weight_curve * loss_curve_all).sum() \
                    + (weight_arg.weight_c_board * c_board_loss + weight_arg.weight_th_board * th_board_loss).sum()
    else:
        loss_back = 0.

    # + (10*first_c_loss).sum()
    return loss_temp,loss_back,spot_size,dist_loss,color_loss,efl,bfl,TOTR,f,opd_loss
# 计算bfl损失
def bfl_cal(bfl,bfl_max,bfl_min,weight_bfl):
    bfl = bfl[:, op_args.wave_idx:op_args.wave_idx+1, 0]
    bfl_loss_max = torch.where(bfl>bfl_max,(bfl-bfl_max),0.)
    bfl_loss_min = torch.where(bfl <bfl_min, (bfl_min - bfl), 0.)
    bfl_loss = weight_bfl*(bfl_loss_max+bfl_loss_min)
    return bfl_loss
# 计算每个镜头的焦距，通常用来计算光焦度标准差的损失
def cal_f(model_cv,model_th,model_rn):
    model_cv = 1/model_cv
    lens_num = model_cv.shape[-1]//2
    f = torch.zeros((model_cv.shape[0],lens_num)).cuda()
    for i in range(lens_num):
        rn = model_rn[:,0:1,2*i+1]
        r1 = model_cv[:,2*i:2*i+1]
        r2 = model_cv[:,2*i+1:2*i+2]
        d = model_th[:,2*i+1:2*i+2]
        f[:,i] = ((rn * r1 * r2) / (rn - 1) / (rn * (r2 - r1) + (rn - 1) * d)).squeeze(-1)
    return f
# 错误梯度归0，防止adam优化器累计nan梯度
def grad_adjust(model,optimizer):
    # 获得曲率和厚度的梯度，并钳制在一定范围内
    cv_grad = torch.where(torch.isnan(model.cv.grad.data), 0., model.cv.grad.data)
    th_grad = torch.where(torch.isnan(model.th.grad.data), 0., model.th.grad.data)
    if op_args.optimiser!='LM':
        if lr_k!=0:
            k_grad = torch.where(torch.isnan(model.k.grad.data), 0., model.k.grad.data)
            optimizer.param_groups[2]['params'][0].grad.data = k_grad * op_args.weight_grd # * k_label_tensor
        if op_args.rn_vara==True:
            rn_grad = torch.where(torch.isnan(model.rn.grad.data), 0., model.rn.grad.data)
            optimizer.param_groups[4]['params'][0].grad.data = rn_grad * op_args.weight_grd
        if model.power_para.grad != None:
            power_grad = torch.where(torch.isnan(model.power_para.grad.data), 0., model.power_para.grad.data)
            optimizer.param_groups[3]['params'][
                0].grad.data = power_grad * op_args.weight_grd

        optimizer.param_groups[0]['params'][0].grad.data = cv_grad * op_args.weight_grd
        optimizer.param_groups[1]['params'][0].grad.data = th_grad * op_args.weight_grd
    else:
        if lr_k!=0:
            k_grad = torch.where(torch.isnan(model.k.grad.data), 0., model.k.grad.data)
            optimizer.param_groups[0]['params'][2].grad.data = k_grad * op_args.weight_grd # * k_label_tensor
        if op_args.rn_vara==True:
            rn_grad = torch.where(torch.isnan(model.rn.grad.data), 0., model.rn.grad.data)
            optimizer.param_groups[0]['params'][4].grad.data = rn_grad * op_args.weight_grd
        if model.power_para.grad != None:
            power_grad = torch.where(torch.isnan(model.power_para.grad.data), 0., model.power_para.grad.data)
            optimizer.param_groups[0]['params'][3].grad.data = power_grad * op_args.weight_grd

        optimizer.param_groups[0]['params'][0].grad.data = cv_grad * op_args.weight_grd
        optimizer.param_groups[0]['params'][1].grad.data = th_grad * op_args.weight_grd

    return optimizer
# 报错阶段性最好结果，并重新记录最优值
def find_and_save_best(loss_person_best,loss_best_all,c_person_best,th_person_best,
                       rn_person_best,person_best_i,c_best_all,th_best_all,rn_best_all,k_person_best,
                       k_best_all,power_person_best,power_best_all):
    c_temp = model.cv.data
    th_temp = model.th.data
    k_temp = model.k.data
    power_temp  = model.power_para.data
    if op_args.method_name=='CURR' and i%100==0:
        c_person_best = c_temp
        th_person_best = th_temp
        loss_person_best = loss_temp
        k_person_best = k_temp
        power_person_best = power_temp
        if op_args.rn_vara == True:
            rn_person_best = torch.where(loss_temp.unsqueeze(-1) < loss_person_best.unsqueeze(-1), rn_temp,rn_person_best)
        else:
            rn_person_best = 0


    c_person_best = torch.where(loss_temp < loss_person_best, c_temp, c_person_best)
    th_person_best = torch.where(loss_temp < loss_person_best, th_temp, th_person_best)
    if lr_k!=0:
        k_person_best = torch.where(loss_temp < loss_person_best, k_temp, k_person_best)
    else:
        k_person_best=0
    if lr_power!=0:
        power_person_best = torch.where(loss_temp.repeat_interleave(lens_args.surf_num, dim=-1).unsqueeze(-1) < loss_person_best.repeat_interleave(lens_args.surf_num,                                                                                             dim=-1).unsqueeze(
                -1), power_temp, power_person_best)
    else:
        power_person_best=0

    if op_args.rn_vara==True:
        rn_person_best = torch.where(loss_temp.unsqueeze(-1) < loss_person_best.unsqueeze(-1), rn_temp, rn_person_best)
        person_best_i = torch.where( (loss_temp < loss_person_best) | (rn_change==True), i + 0.0,
                                person_best_i)  # 记录每个粒子最小值下降的最近的代数，用来判断该粒子多久没下降过了,如果折射率改变了，也记录
    else:
        rn_person_best=0
        person_best_i = torch.where( (loss_temp < loss_person_best), i + 0.0,
                                person_best_i)  # 记录每个粒子最小值下降的最近的代数，用来判断该粒子多久没下降过了,如果折射率改变了，也记录
    loss_person_best = (torch.where(loss_temp < loss_person_best, loss_temp, loss_person_best)).data
    loss_person_best_median = torch.median(loss_person_best)
    # eff_part = c_person_best[(loss_person_best<0.4).squeeze(1),:].data.cpu().numpy()

    # with torch.no_grad():
    judge_guidence = torch.where(((i - person_best_i > op_args.stop_time) & (loss_person_best > loss_person_best_median) & (
                loss_person_best > op_args.eff_v) ), 1., 0.)
        # print('eff:', torch.sum((loss_person_best < op_args.eff_v)))
    # judge_guidence = torch.where(((i - person_best_i > op_args.stop_time) & (loss_person_best > loss_person_best_median) ), 1., 0.)
    print('eff num:', torch.sum((loss_person_best < op_args.eff_v)))
    idx_best = torch.argmin(loss_temp, dim=0)

    c_best = c_temp[idx_best]  # 更新前的最好值
    th_best = th_temp[idx_best]
    if lr_k!=0:
        k_best = k_temp[idx_best]
    else:
        k_best=0
    if lr_power!=0:
        power_best = power_temp[idx_best]
    else:
        power_best = 0.
    if op_args.rn_vara==True:
        rn_best = rn_temp[idx_best]
    else:
        rn_best = 0
    if op_args.method_name=='CURR' and i%100==0:
        c_best_all = c_best.data  # [1,10]
        th_best_all = th_best.data
        if lr_k!=0:
            k_best_all = k_best.data
        else:
            k_best_all=0
        if lr_power!=0:
            power_best_all = power_best.data
        else:
            power_best_all=0
        if op_args.rn_vara==True:
            rn_best_all = rn_best.data
        else:
            rn_best_all = 0
        print('best cv: ')
        loss_best_all = loss_temp[idx_best].item()

    if loss_temp[idx_best].item() < loss_best_all:  # or i%100==0:
        c_best_all = c_best.data  # [1,10]
        th_best_all = th_best.data
        if lr_k!=0:
            k_best_all = k_best.data
        else:
            k_best_all=0
        if lr_power!=0:
            power_best_all = power_temp[idx_best]
        else:
            power_best_all = 0
        if op_args.rn_vara==True:
            rn_best_all = rn_best.data
        else:
            rn_best_all = 0
        print('best cv: ')
        loss_best_all = loss_temp[idx_best].item()
    return c_person_best,c_best,c_best_all,th_person_best,th_best,th_best_all,rn_person_best,rn_best,rn_best_all,\
           idx_best,person_best_i,loss_person_best,loss_person_best_median,loss_best_all\
        ,judge_guidence,k_person_best,k_best,k_best_all,power_person_best,power_best_all,power_best
# 保存镜头全局优化结果
def save_lens_opt_result(loss_list,loss_sum, c_best, th_best, rn_person_best, rn_best_all,
                                         c_best_all, th_best_all,k_best,k_best_all,power_best,power_best_all):
    loss_list_for_save = np.array(loss_list)
    with open('../data/loss_pkl/'+op_args.file_name+'/'+op_args.file_name+'_'+op_args.experiment_name+'.pkl', 'wb') as f:
        pickle.dump(loss_list_for_save, f)
    loss_list_for_save = np.array(loss_sum)
    with open('../data/loss_pkl/'+op_args.file_name+'/'+op_args.file_name+'_'+op_args.experiment_name+'_loss_sum.pkl', 'wb') as f:
        pickle.dump(loss_list_for_save, f)
    print('last op_args.epoch c: ', c_best)
    print('last op_args.epoch th: ', th_best)
    # print((spot_size *).tolist())
    # save_data_as_excel(spot_size , 'final_spot', 1)
    if op_args.rn_vara==True:
        rn_temp, material_idx_best_all = find_best_material(rn_database, rn_person_best, rn_min_list, rn_max_list)
        rn_temp, material_idx_best = find_best_material(rn_database, rn_best_all, rn_min_list, rn_max_list)

        best_glass_name = []
        for glass in range(material_idx_best.shape[-1]):
            best_glass_name.append(glass_name[int(material_idx_best[0, glass].item())])
        print('best glass',best_glass_name)
        model_test.rn.data = rn_best_all
    model_test.cv.data = c_best_all
    # model.cv.data = 1/torch.tensor([  11.9580, -182.0207,   12.0033,  779.0150,  -10.0000,   54.2819],dtype = torch.float64)
    model_test.th.data = th_best_all
    if lr_k!=0:
        model_test.k.data = k_best_all
    if lr_power!=0:
        model_test.power_para.data = power_best_all
    # model_test.power_para.data = power_best_all
    with torch.no_grad():
        # forward_cal
        spot_size, bfl, efl, y_field, loss_intersect, y_surf_height_loss, color_loss, loss_curve_all,opd = model_test(
            x_init, y_init, z_init, l_init, m_init, n_init)
    R_best = 1 / c_best_all
    print(loss_best_all)
    print('last spot size :', spot_size[0] )
    print('R:', 1 / c_best_all )
    print('th: ', th_best_all )
    if lr_k==0:
        th_save = torch.cat((th_best_all , bfl[:,op_args.wave_idx:op_args.wave_idx+1,0] , R_best ), dim=1)
    else:
        th_save = torch.cat((th_best_all * efl_scale, bfl[:, op_args.wave_idx:op_args.wave_idx+1, 0] * efl_scale, R_best * efl_scale, k_best_all),
                            dim=1)
    if op_args.rn_vara == True:
        th_save = torch.cat((th_save,material_idx_best),dim=1)


    if lr_power!=0:
        power_best_all = power_best_all[0]
        for i in range(power_best_all.shape[-1]):
            power_best_all[:, i] = power_best_all[:, i]
        save_data_as_excel(power_best_all, op_args.file_name+'/'+op_args.file_name+'_power_'+op_args.experiment_name, 0)
        with open('../data/datatxt/'+op_args.file_name+'/power/'+op_args.file_name+'_power_all_' + op_args.experiment_name + '.pkl', 'wb') as f:
            pickle.dump(power_person_best.data.detach().cpu().numpy(), f)
    print('th_save:', th_save)

    # power_best_all = power_best_all[0]
    # for i in range(power_best_all.shape[-1]):
    #     power_best_all[:, i] = power_best_all[:, i] * (efl_scale ** (-(2 * i + 3)))
    try:
        save_data_as_excel(th_save, op_args.file_name+'/'+op_args.file_name+'_'+op_args.experiment_name, 1)
        # save_data_as_excel(power_best_all, 'Ultraviolet_power', 0)
    except:
        save_data_as_excel(th_save, op_args.file_name+'/'+op_args.file_name+'_'+op_args.experiment_name, 1)
        # save_data_as_excel(power_best_all, 'Ultraviolet_power_1', 0)


    # 保存所有的系统参数用来挑选c,th.k,power
    if lr_k!=0:
        data_for_save = torch.cat((th_person_best,1/c_person_best,k_person_best,loss_person_best),dim = -1)
    else:
        data_for_save = torch.cat((th_person_best,1/c_person_best,loss_person_best),dim = -1)
    if op_args.rn_vara == True:
        data_for_save = torch.cat((data_for_save,material_idx_best_all),dim=-1)

    save_data_as_excel(data_for_save, op_args.file_name+'/'+op_args.file_name+'_all_' + op_args.experiment_name+'', 1)

# ****************************************************************************************************************************
a=1

def test_code():
    data = read_excel('../data/' + op_args.file_name + '.xlsx')
    c = 1 / data[op_args.data_idx, lens_args.surf_num:2 * lens_args.surf_num].unsqueeze(0).repeat(op_args.N, 1)
    th = data[op_args.data_idx, :lens_args.surf_num].unsqueeze(0).repeat(op_args.N, 1)
    model = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, th_label,
                                      omit=lens_args.power_omit, field_num=op_args.field_num,
                                      stop_idx=lens_args.stop_idx,
                                      wave_number=lens_args.wave_number).cuda()
    model.cv.data = c.cuda()
    model.th.data = th.cuda()
    spot_size, bfl, efl, y_field, loss_intersect, loss_angle, color_loss, loss_curve_all,opd = model(
            x_init, y_init, z_init, l_init, m_init, n_init,op_init=op_init)
    print(spot_size)

# ****************************************************************************************************************************
a=1
# 设置随机种子
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)             # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)        # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True  # NVD深度训练加速算法库
# 从excel读取文件
def read_excel(file_path, T=1,norm = 1):
    data = pd.read_excel(file_path)
    data = data.values
    data = torch.from_numpy(data)
    data = data
    if T==1:
        data = data.transpose(0,1)
    return data
# 保存二维数据进excel
def save_data_as_excel(data,name,T = 0):
    if T ==1:
        data = data.transpose(0,1)
    data = data.detach().cpu().numpy()
    np.savetxt('../data/datatxt/'+name+'.txt',data,fmt = '%e')
    # df = pd.DataFrame(data)
    # df.to_excel('data/'+name+".xlsx", index=False)
# 暂时没用
# ****************************************************************************************************************************
a=1


zero_1 = torch.tensor([0]).cuda()
def plot_lens_code(data_file_name,only_max_min=True,old_data=False):



    if not os.path.exists('../data/plot_lens_fig/'+op_args.file_name+'/'+data_file_name):
        os.makedirs('../data/plot_lens_fig/'+op_args.file_name+'/'+data_file_name)
        print(f"Directory '{'../data/plot_lens_fig/'+op_args.file_name+'/'+data_file_name}' created.")
    else:
        print(f"Directory '{'../data/plot_lens_fig/'+op_args.file_name+'/'+data_file_name}' already exists.")
    data = torch.from_numpy(np.loadtxt('../data/plot_lens_data/' + data_file_name + '.txt')).cuda().transpose(0, 1)
    if op_args.file_name=='6p':
        with open('../data/plot_lens_data/'+data_file_name+'_power.pkl', 'rb') as f:
            power_para = pickle.load(f)
        power_para = torch.from_numpy(power_para).cuda()
    if only_max_min==True:
        mask = (data[:, -1] < op_args.eff_v)
        data = data[mask]
        if op_args.file_name == '6p':
            power_para = power_para[mask, :, :]
        print('eff num: ', mask.sum())
        if mask.sum()>=data.shape[-1]:
            min_mask = torch.min(data, dim=0).indices
            max_mask = torch.max(data, dim=0).indices
            plot_mask = torch.cat((min_mask, max_mask), dim=0)
            plot_mask = (data[:,1]<2.5)
            # 仅画单个变量最大或者最小的
            data = data[plot_mask]
            if op_args.file_name == '6p':
                power_para = power_para[plot_mask, :, :]
        else:
            print('eff num is no enough to show max/min of each variable')
            pass
    # if op_args.stop_idx!=1 and op_args.stop_fix==False:
    c = 1 / data[:, lens_args.surf_num:2 * lens_args.surf_num]
    th = data[:, :lens_args.surf_num]
    if old_data == True:
        th_stop = -torch.sum(th[:,:lens_args.stop_idx-1],dim=-1,keepdim=True)
        th_stop_surf = (th[:,lens_args.stop_idx-2] + th[:,lens_args.stop_idx-1] ).unsqueeze(-1)
        th = torch.cat((th_stop,th[:,:lens_args.stop_idx-2],th_stop_surf,th[:,lens_args.stop_idx:]),dim=-1)
    op_args.N = c.shape[0]
    model = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, th_label,
                                      omit=lens_args.power_omit, field_num=op_args.field_num,
                                      stop_idx=lens_args.stop_idx,
                                      wave_number=lens_args.wave_number).cuda()

    if op_args.file_name=='6p':
        model.power_para.data = power_para.cuda()
        k = data[:,2 * lens_args.surf_num:-1]
        model.k.data = k.cuda()
    model.cv.data = c.cuda()
    model.th.data = th.cuda()
    model.plot_model_lens(op_args,data_file_name)
    sys.exit()

# 模型函数
class spot_size_layer_multiwave(nn.Module):
    def __init__(self,sur_num = 6,lenth_num = 6,power_num = 3,th_label=[4.8,4.8,3.188,3.188,6.5,6.5],
                 omit = [1,1,1,1,1,1,1,1,1,1,1],field_num = 3,stop_idx=1,wave_number = 3):
        super(spot_size_layer_multiwave, self).__init__()
        cv = torch.zeros((sur_num))
        #cv = torch.rand((sur_num), dtype=torch.float64)*0.2-0.1
        # th = torch.rand((lenth_num), dtype=torch.float64)*(th_max-th_min)+th_min
        th = torch.ones((lenth_num)) *0.5
        # rn = torch.rand((3,lenth_num+1),dtype=torch.float64)*0.2+1.48

        rn = lens_args.rn_list
        rn = rn.repeat(1,rn_repeat_num,1)
        k = torch.zeros( (op_args.N,sur_num))
        power_para = torch.zeros( (op_args.N,sur_num,power_num))

        tilt_l = torch.zeros((1,1, sur_num))
        tilt_m = torch.zeros((1,1, sur_num))

        decenter_x = torch.zeros((1,1, sur_num))
        decenter_y =torch.zeros((1,1, sur_num))
        decenter_z = torch.zeros((1,1, sur_num))





        # self.tilt_l = nn.Parameter(tilt_l)
        # self.tilt_m = nn.Parameter(tilt_m)
        # self.decenter_x = nn.Parameter(decenter_x)
        # self.decenter_y = nn.Parameter(decenter_y)
        # self.decenter_z = nn.Parameter(decenter_z)

        self.tilt_l = tilt_l.cuda()
        self.tilt_m = tilt_m.cuda()
        self.decenter_x = decenter_x.cuda()
        self.decenter_y = decenter_y.cuda()
        self.decenter_z = decenter_z.cuda()

        # 对于两个方向同时倾斜出现的错误，可能原因是倾斜的方向向量平方和没有归一导致
        # self.decenter_x.data[0, 0, 0] = 0.
        # self.decenter_y.data[0, 0, 3] = 0.
        #
        # value = 0
        # self.tilt_l.data[0, 0, 3] =0/180 * math.pi # m=-x, l=y
        # self.tilt_m.data[0, 0, 3] = 2*value / 180 * math.pi

        # 将空气折射率先初始化为1

        # th = torch.tensor([0,           2       ,5.26   ,   1.25   ,   4.69   , 2.25  ],dtype=torch.float64)
        # th[0]=0;th[1]=2;th[3]=1.25;th[5]=2.25
        #cv = torch.tensor([1 / 21.48138, 1 / -124.1, 1 / -19.1, 1 / 22, 1 / 648.9, 1 / -16.7], dtype=torch.float64)
        self.field_num = op_args.field_num
        self.omit = omit
        self.th_label = th_label
        self.th = nn.Parameter(th)
        self.cv = nn.Parameter(cv)
        if op_args.rn_vara==True:
            self.rn = nn.Parameter(rn)
        else:
            self.rn = rn.cuda()
        if lr_k!=0:
            self.k = nn.Parameter(k)
        else:
            self.k = k.cuda()
        if lr_power!=0:
            self.power_para = nn.Parameter(power_para)  # nn.Parameter(power_para)
        else:
            self.power_para = power_para.cuda()
        self.stop_idx = stop_idx
        self.wave_number=wave_number
    def cal_th_for_trace(self,return_beta=False):
        # This function is used to calculate the thickness at each interval from the entry pupil
        # to the last side of the lens for forward propagation
        if op_args.stop_fix ==True:
            # Take Cooke for example,
            # self.stop_idx = 4,self.th will be[glass,air,stop,stop,air,glass]
            # self.stop_idx = 3,self.th will be[glass,stop,stop,glass,air,glass]
            # self.stop_idx = 1,self.th will be[stop,glass,air,glass,air,glass]
            # the th_for_ray_trace should always be [entrance2first_surf,glass,air,glass,air,glass]
            if self.stop_idx == 1:
                # self.th.data[:, 0] = 0.
                th_for_ray_trace = self.th + self.decenter_z[:,0,:]
                # th_for_ray_trace[:,0].data=torch.tensor([0.]).cuda()

            else:
                # 有的时候需要返回入瞳和光阑的缩放比例关系
                if return_beta==False:
                    th_for_ray_trace = find_entrance(self.cv, self.th , self.rn,self.stop_idx)
                else:
                    th_for_ray_trace,beta =  find_entrance(self.cv, self.th , self.rn,self.stop_idx,True)
        else:
            # if stop_fix==False, the self.th will be [stop_pos,glass,air,glass,air,glass]
            # and we need to find where the entrance is
            th_for_ray_trace = find_entrance_with_stop_float(self.cv, self.th, self.rn, self.stop_idx)
        if return_beta==False:
            return th_for_ray_trace
        else:
            return th_for_ray_trace,beta

    def cal_efl_bfl(self, th_for_ray_trace):
        efl, bfl = cal_para_focus_by_newton_multi_sys(self.cv, th_for_ray_trace, self.rn, self.k, self.omit,
                                                      self.power_para)
        bfl = torch.where(torch.isnan(bfl.data), 200., bfl)

        return efl, bfl


    def forward_efl(self, x_out, y_out, z_out, l, m, n, only_bfl=False, rand_zero=-1, color_bfl=True):
        th_for_ray_trace = self.cal_th_for_trace()
        efl, bfl = self.cal_efl_bfl(th_for_ray_trace)
        return bfl.cuda(), efl.cuda()

    def cal_exit(self):
        exit_pos = find_exit(self.cv, self.th, self.rn, self.stop_idx)
        return exit_pos

    def from_last_surf2exit(self,x_out, y_out, z_out, l, m, n,opl,bfl,efl):
        # 计算出瞳位置
        exit_pos = (self.cal_exit()).unsqueeze(-1).unsqueeze(-1)
        nm =int(z_out.shape[-1]/op_args.field_num)
        # 得到系统数量和波段数量
        op_args.N = x_out.shape[0]; rn_num = x_out.shape[1]
        # 将数据变形成【系统数，波段数，视场数，光线数】，原本视场和光线是在一维
        opl = opl.view((op_args.N, rn_num, op_args.field_num, nm))
        x_out = x_out.view((op_args.N,rn_num,op_args.field_num,nm))
        y_out = y_out.view((op_args.N,rn_num,op_args.field_num,nm))
        z_out = z_out.view((op_args.N,rn_num,op_args.field_num,nm))
        l = l.view((op_args.N,rn_num,op_args.field_num,nm))
        m = m.view((op_args.N,rn_num,op_args.field_num,nm))
        n = n.view((op_args.N,rn_num,op_args.field_num,nm))
        # 追迹到出瞳，只计算交点，不计算折射
        x_out, y_out, z_out,op = \
            ray_trace_newton_only_point(x_out, y_out, z_out, l, m, n, 0, 0, exit_pos, zero_3_dim, zero_3_dim, zero_3_dim, 0, 0., self.rn[:,:, -1:]
                             , 1, 0,cal_op=True)
        # 得到出瞳面上，各主光线的位置和角度
        opl = opl+op
        idx_center = nm//2

        # 计算出瞳到像面距离
        z_dist = bfl.unsqueeze(-1) - exit_pos

        opd = self.calopd(z_dist,x_out,y_out,z_out,l, m, n,idx_center,opl)

        return opd

    def calopd(self,z_dist,x_out,y_out,z_out,l, m, n,idx_center,opl):
        x_pos_of_chief_ray = x_out[:, :, :, idx_center:idx_center + 1]
        y_pos_of_chief_ray = y_out[:, :, :, idx_center:idx_center + 1]
        alpah_pos_of_chief_ray = torch.atan((l / n)[:, :, :, idx_center:idx_center + 1])
        beta_pos_of_chief_ray = torch.asin(m[:, :, :, idx_center:idx_center + 1])
        # beta_pos_of_chief_ray = torch.atan((m/n)[:, :, :, idx_center:idx_center + 1])
        gama_pos_of_chief_ray = torch.asin(n[:, :, :, idx_center:idx_center + 1])
        n_pos_of_chief_ray = n[:, :, :, idx_center:idx_center + 1]
        # m_pos_of_chief_ray = (1 - m[:, :, :, idx_center:idx_center + 1] ** 2) ** 0.5
        # l_pos_of_chief_ray = (1 - l[:, :, :, idx_center:idx_center + 1]**2)**0.5
        # 现在是追迹到了出瞳所在平面，得到平面上的主光线的位置和出射角度,需要从当前位置，追迹点到下一个球面
        # 每个视场，需要得到一个自己的理想波面的曲率，也就是每个视场主光线在光瞳面处的位置及角度，到达像面的距离0
        cv_of_every_fov = 1 / (z_dist / n_pos_of_chief_ray)

        # beta对应acos(n) alpha对应acos(l)  gama对应acos(m)
        # 追迹到理想波像面，只计算交点，不计算折射
        x_out, y_out, z_out, op = \
            ray_trace_newton_only_point(x_out, y_out, z_out, l, m, n, x_pos_of_chief_ray, y_pos_of_chief_ray, 0,
                                        alpah_pos_of_chief_ray,
                                        beta_pos_of_chief_ray, zero_1, 0, cv_of_every_fov, self.rn[:, :, -1:], 1, 0,
                                        cal_op=True)
        opl = opl + op
        opd = (opl[:, :, :, idx_center:idx_center + 1] - opl) / wave_sys
        return opd


    def entrance2last_lensurf(self, x_init, y_init, z_init, l_init, m_init, n_init,op_init = 0):
        th_for_ray_trace = self.cal_th_for_trace()
        efl, bfl = self.cal_efl_bfl(th_for_ray_trace)

        loss_intersect = 0
        y_surf_height_loss = 0
        # 先追迹到玻璃的最后一面
        if op_args.cal_opd == False:
            op_init = 0
        loss_angle = 0.
        loss_curve_all = 0.
        opl = op_init
        x_out, y_out, z_out, l, m, n = x_init.clone(), y_init.clone(), z_init.clone(), l_init.clone(), m_init.clone(), n_init.clone()
        # x_out, y_out, z_out, l, m, n = x_init , y_init , z_init , l_init , m_init , n_init
        # l_init[0,0,5] = 0.
        for i in range(self.cv.shape[1]):
            x_out, y_out, z_out, l, m, n, op, loss_intersect_layer, loss_refract_layer, loss_curve = \
                ray_trace_newton(x_out, y_out, z_out, l, m, n, self.decenter_x[:, :, i:i + 1],
                                 self.decenter_y[:, :, i:i + 1], th_for_ray_trace[:, i:i + 1].unsqueeze(-1),
                                 self.tilt_l[:, :, i:i + 1],
                                 self.tilt_m[:, :, i:i + 1], zero_1,
                                 self.k[:, i:i + 1].unsqueeze(-1), self.cv[:, i:i + 1].unsqueeze(-1),
                                 self.rn[:, :, i:i + 1],
                                 self.rn[:, :, i + 1:i + 2], self.omit[i], self.power_para[:, i:i + 1, :],
                                 self.th_label[i], cal_op=op_args.cal_opd,edge=lens_args.edge_min)

            n_nonan = torch.where(torch.isnan(n.data), 200., n)
            n_nonan = (torch.min(n_nonan, dim=-1)).values
            n_nonan = (torch.min(n_nonan, dim=-1, keepdim=True)).values
            angle_limit = torch.where((n_nonan - 0.7) < 0., torch.abs(n_nonan - 0.7), 0.)
            loss_angle = loss_angle + angle_limit
            loss_curve_all = loss_curve_all + loss_curve
            if i > 0:
                loss_intersect = (loss_intersect + loss_intersect_layer)  # 避免曲面出现交叉
            if op_args.cal_opd == True:
                opl = opl + op
        return  x_out, y_out, z_out, l, m, n, efl,bfl,opl,loss_angle,loss_curve_all,loss_intersect

    def last_lens2_image(self, x_out, y_out, z_out, l, m, n,bfl):
        x_out, y_out, z_out, l, m, n, op, _, _, _ = \
            ray_trace_newton(x_out, y_out, z_out, l, m, n, 0, 0, bfl[:, op_args.wave_idx:op_args.wave_idx+1, :], zero_3_dim, zero_3_dim, zero_1, 0, 0., self.rn[:,:, -1:],
                             self.rn[:,:, -1:], 1, 0,cal_op=op_args.cal_opd)
        nm_each_field = int(x_out.shape[-1]/op_args.field_num)
        height_field_center = y_out.view(y_out.shape[0],y_out.shape[1],op_args.field_num,-1)[:,:,:,nm_each_field//2]
        color_y = torch.sum(torch.abs(height_field_center[:,0] - height_field_center[:,-1]),dim=-1,keepdim=True)

        spotsize, avg_y_list ,avg_x_list,y_real_mw,x_real_mw= cal_mlzi_spot_size(x_out, y_out, self.field_num)

        # print(spotsize)
        color_loss = torch.abs(bfl[:, -1, :] - bfl[:, 0, :])
        color_loss = color_loss+color_y
        return spotsize, avg_y_list,color_loss

    def ray_aming_entrance2stop(self):
        start = time.time()
        with torch.no_grad():

            # self.th.data = self.th.repeat(2,1)
            # self.cv.data = self.cv.repeat(2, 1)
            N = self.th.shape[0]
            # 当视场大，误差非常大时，以入瞳为采样点，会出现不经过光阑对应点的现象，并且最终的光程差等均出现较大偏差
            # 光线瞄准思路：cal_efl_bfl_th_for_trace函数可以得到入瞳位置，并且可以选择返回入瞳和光阑的尺寸比例，即光阑实际尺寸。
            # 知道实际尺寸后，采样光阑的 上左中下右 五个点，面向前一个表面的最大光线允许处，一个超大密度光线追迹
            # 假如对于降敏的情况，需要大密度视场采样，那么光线瞄准估计得4*100*100这个强度。
            # 前向追迹后，得到第一个表面前的出射光线信息，从中筛选(l,m,n)满足需要采样视场的光线，其在入瞳位置的 xy信息，作为光线追迹的真正采样点
            th_for_ray_trace,entrance_stop_beta = self.cal_th_for_trace(True)
            stop_size = (  lens_args.half_apture_max / entrance_stop_beta)
            th_before_stop = torch.cat((th_for_ray_trace[:,:self.stop_idx-1],self.th[:,self.stop_idx-2:self.stop_idx-1]),dim=-1)
            th_before_stop = torch.flip(th_before_stop,dims=[-1])
            # 要增加最后一个面的曲率为0，为入瞳所在的面,所以这里多拿一个面，然后设成0
            cv_before_stop = -torch.flip(self.cv[:,:self.stop_idx-1],dims=[-1])
            zero_column = torch.zeros(N, 1).cuda()

            cv_before_stop = torch.cat((cv_before_stop,zero_column),dim=-1)
            # 折射率也要多拿一个，补成1
            rn_before_stop = torch.flip(self.rn[:, op_args.wave_idx:op_args.wave_idx+1, :self.stop_idx], dims=[-1])
            zero_column = zero_column.unsqueeze(-1)
            rn_before_stop = torch.cat((rn_before_stop,zero_column[0:1]+1),dim=-1)
            # 偏心数据这些也要多拿一个，补成0
            if op_args.desensitive==False:
                zero_column = zero_column[0:1]
            decenter_x_before_stop = torch.flip(self.decenter_x[:,:,:self.stop_idx-1],dims=[-1])
            decenter_x_before_stop = torch.cat((decenter_x_before_stop,zero_column),dim=-1)
            decenter_y_before_stop = torch.flip(self.decenter_y[:, :,:self.stop_idx-1], dims=[-1])
            decenter_y_before_stop = torch.cat((decenter_y_before_stop,zero_column),dim=-1)
            tilt_l_before_stop = torch.flip(self.tilt_l[:, :,:self.stop_idx-1], dims=[-1])
            tilt_l_before_stop = torch.cat((tilt_l_before_stop,zero_column),dim=-1)
            tilt_m_before_stop = torch.flip(self.tilt_m[:, :,:self.stop_idx-1], dims=[-1])
            tilt_m_before_stop = torch.cat((tilt_m_before_stop,zero_column),dim=-1)
            # 使用光阑尺寸，前面的th,cv(用来算往前的最大尺寸)

            # 从光阑指定的几个点反向传播大密度光线，利用算的lmn和理想正向的lmn匹配，找到最匹配的这些lmn对应的在光阑位置处的xy，作为后面牛顿迭代的起点
            x_out,y_out,z_out,l, m, n,stop_sample_num,stop_cord_x,stop_cord_y = \
                ray_aming_stop_init_ray(stop_size, th_before_stop,op_args.sample_delta, sample_num=201)



            # 得到光阑投射到前一透镜的光线后，进行光线追迹
            # th应该将第一个替换成0
            th_before_stop[:,0]  = 0.
            # x_out, y_out, z_out, l, m, n = x_aming_init.data, y_aming_init.data, z_aming_init.data, \
            #                                L_aming_init.data, M_aming_init.data, N_aming_init.data
            # x_out, y_out, z_out, l, m, n = x_init , y_init , z_init , l_init , m_init , n_init
            for i in range(cv_before_stop.shape[1]):
                x_out, y_out, z_out, l, m, n, _, _, _, _ = \
                    ray_trace_newton(x_out, y_out, z_out, l, m, n, decenter_x_before_stop[:, :, i:i + 1],
                                     decenter_y_before_stop[:, :, i:i + 1], th_before_stop[:, i:i + 1].unsqueeze(-1),
                                     tilt_l_before_stop[:, :, i:i + 1],
                                     tilt_m_before_stop[:, :, i:i + 1], zero_1,
                                     0., cv_before_stop[:, i:i + 1].unsqueeze(-1),
                                     rn_before_stop[:, :, i:i + 1],
                                     rn_before_stop[:, :, i + 1:i + 2], 1, 0.,
                                     self.th_label[i], cal_op=False)


            # 至此，得到的l,m,n是入瞳处的光线角度，从中选出满足条件(最接近物点发出的，到达入瞳中心的方向的lmn)的lmn
            # 需要先计算，希望采样的视场的光线角度。
            _, _,_, l_init, m_init, n_init, _, _,_ = grid_data_of_staring_point_para(
                lens_args.half_apture_max, field_y,field_x= field_x,
                sample_delta=op_args.sample_delta)
            l_init = l_init.repeat(N,1,1).reshape(N,op_args.field_num,-1).permute(0,2,1).unsqueeze(-1) # 得到的结果是[Batch,sample(5), op_args.field_num]
            m_init = m_init.repeat(N,1,1).reshape(N, op_args.field_num, -1).permute(0,2,1).unsqueeze(-1)
            n_init = n_init.repeat(N,1,1).reshape(N, op_args.field_num, -1).permute(0,2,1).unsqueeze(-1)
            lmn_target = torch.cat((l_init,m_init,n_init),dim=-1)

            # 由于这里追迹是反着来的，l和m是不是得加负号
            l = -l.reshape(N,stop_sample_num,-1).unsqueeze(-1)
            m = -m.reshape(N, stop_sample_num, -1).unsqueeze(-1)
            n = n.reshape(N, stop_sample_num, -1).unsqueeze(-1)
            x_out = x_out.reshape(N,stop_sample_num,-1)
            y_out = y_out.reshape(N, stop_sample_num, -1)
            lmn_database = torch.cat((l,m,n),dim=-1)

            # 计算欧氏距离

            # distances =  -Cos_sim_2patch(lmn_target,lmn_database)
            distances = torch.cdist(lmn_target, lmn_database)
            # 找到最近的stop_sample_num个值的索引
            topk_values, topk_indices = torch.topk(distances, k=3, dim=-1, largest=False)
            # topk_indices = topk_indices.squeeze(-1)
            # x_target = torch.gather(x_out.unsqueeze(-1), dim=-1, index=topk_indices).permute(0,2,1)
            # y_target = torch.gather(y_out.unsqueeze(-1), dim=-1, index=topk_indices).permute(0,2,1)
            print(topk_values.sum())
            x_target = torch.mean(torch.gather(x_out.unsqueeze(-2).repeat(1,1,op_args.field_num,1), dim=-1, index=topk_indices),dim=-1).permute(0, 2, 1)
            y_target = torch.mean(torch.gather(y_out.unsqueeze(-2).repeat(1,1,op_args.field_num,1), dim=-1, index=topk_indices),dim=-1).permute(0, 2, 1)

            end = time.time()

        l_init = l_init[:, :, :, 0].permute(0, 2, 1).reshape(N,1,-1)
        m_init = m_init[:, :, :, 0].permute(0, 2, 1).reshape(N,1,-1)
        n_init = n_init[:, :, :, 0].permute(0, 2, 1).reshape(N,1,-1)
        # 得到xy target后，将这些值xy设置为变量,沿着光学系统正向追迹到光阑，要求光阑处的xy坐标和理应的坐标相等，
        # 需要一个函数，这个函数输入为x_init,y_init(优化起点)，输入系统参数，光阑直径(优化目标)
        t1 =time.time()
        x_init,y_init = self.ray_aming_entrance2stop_iter(th_for_ray_trace,x_target,y_target,stop_size,l_init,m_init,n_init,stop_cord_x,stop_cord_y)
        t2 = time.time()
        # print('iter time:',t2-t1)
        print('ray aming time: ', end - start)

        z_init = torch.zeros(x_init.shape).cuda()
        return x_init,y_init,z_init,l_init,m_init,n_init


            # 对于model.cv th rn,需要从中选出光阑以前的那些透镜的信息，并翻转，构成前向追迹的参数
            # 计算光阑前面透镜的最大x,y,生成meshgrid，计算

    def ray_aming_entrance2stop_iter(self,th_for_ray_trace,x_init,y_init,stop_size,l_init,m_init,n_init,stop_cord_x,stop_cord_y):
        # 得到xy target后，将这些值xy设置为变量,沿着光学系统正向追迹到光阑，要求光阑处的xy坐标和理应的坐标相等，
        # l,m,n是追迹的第一个表面的值，这里直接弄进来就不用再算了
        # 需要一个函数，这个函数输入为x_init,y_init(优化起点)，输入系统参数，光阑直径(优化目标)
        # 需要先明确优化目标
        with torch.no_grad():
            N = x_init.shape[0]
            zero_temp = torch.zeros(stop_size.shape).cuda()
            stop_cord_x = stop_cord_x.permute(0,2,1).repeat(1,op_args.field_num,1).reshape(N,1,-1)
            stop_cord_y = stop_cord_y.permute(0,2,1).repeat(1,op_args.field_num,1).reshape(N,1,-1)
            # 将输入变形一下，变成可光线追迹的样子
            x_init = x_init.reshape(N,1,-1) # contiguous
            y_init = y_init.reshape(N, 1, -1)
            z_init = torch.zeros(y_init.shape).cuda()
            # 确定曲率
            cv_before_stop = self.cv[:, :lens_args.stop_idx].clone()
            cv_before_stop[:,-1] = 0.
            th_before_stop = torch.cat((th_for_ray_trace[:, : 1],self.th[:,:self.stop_idx-1].clone()),dim=-1)
            rn_before_stop = self.rn[:, op_args.wave_idx:op_args.wave_idx+1, :lens_args.stop_idx+1].clone()
            rn_before_stop[:,:,-1]  = 1
            decenter_x_before_stop = (self.decenter_x[:, :, :self.stop_idx]).clone()
            decenter_x_before_stop[:,:,-1]  = 0.
            decenter_y_before_stop = (self.decenter_y[:, :, :self.stop_idx]).clone()
            decenter_y_before_stop[:,:,-1]  = 0.
            tilt_l_before_stop = (self.tilt_l[:, :, :self.stop_idx]).clone()
            tilt_l_before_stop[:,:,-1]  = 0.
            tilt_m_before_stop = (self.tilt_m[:, :, :self.stop_idx]).clone()
            tilt_m_before_stop[:,:,-1]  = 0.

        # 设置待优化量为变量

        lr_rayaming = 0.1
        # 迭代优化过程：
        params_dict = [
            {'params': x_init, 'lr': lr_rayaming}, {'params': y_init, 'lr': lr_rayaming}
        ]
        with torch.enable_grad():
            # 初始的y_init反了
            for epoch in range(10):
                # y_init[0][0][5] = 3.2474869709E+00		 # 2.0175810833E+00
                # x_init[0][0][5] =1.5935100360E+00
                x_init = nn.Parameter(x_init)
                y_init = nn.Parameter(y_init)
                x_out, y_out, z_out, l, m, n = x_init,y_init,z_init,l_init,m_init,n_init
                for i in range(cv_before_stop.shape[1]):
                    x_out, y_out, z_out, l, m, n, _, _, _, _ = \
                        ray_trace_newton(x_out, y_out, z_out, l, m, n, decenter_x_before_stop[:, :, i:i + 1],
                                         decenter_y_before_stop[:, :, i:i + 1], th_before_stop[:, i:i + 1].unsqueeze(-1),
                                         tilt_l_before_stop[:, :, i:i + 1],
                                         tilt_m_before_stop[:, :, i:i + 1], zero_1,
                                         0., cv_before_stop[:, i:i + 1].unsqueeze(-1),
                                         rn_before_stop[:, :, i:i + 1],
                                         rn_before_stop[:, :, i + 1:i + 2], 1, 0.,
                                         self.th_label[i], cal_op=False)
                loss  = (torch.abs(stop_cord_x - x_out) +  torch.abs(stop_cord_y - y_out)).sum()
                if loss<1e-8:
                    break
                # print('epoch: ',epoch,' ray aming loss: ',loss)
                loss.backward()
                with torch.no_grad():
                    F_x = torch.abs(stop_cord_x - x_out)
                    step_x = F_x / (x_init.grad + 1e-8)
                    F_y = torch.abs(stop_cord_y - y_out)
                    step_y = F_y / (y_init.grad + 1e-8)
                    if (torch.abs(step_y) + torch.abs(step_x)).sum()<1e-8:
                        break
                    x_init = x_init - step_x
                    y_init = y_init - step_y
                    # print(loss)

                x_init.grad = None
                y_init.grad = None
        # print('ray aming iter num: ',epoch)
        return x_init.data,y_init.data

    def forward(self, x_init, y_init, z_init, l_init, m_init, n_init,op_init = 0):
        # if stop is not fix:



        if op_args.ray_aming==True:
            x_init, y_init, z_init, l_init, m_init, n_init = self.ray_aming_entrance2stop()
            op_init = entrance_gird2opd_init(x_init, y_init, n_init, None, None, field_x, field_y, obj=-10e7)
        # with torch.enable_grad():
        x_out, y_out, z_out, l, m, n,efl,bfl, opl, loss_angle, loss_curve_all,loss_intersect = self.entrance2last_lensurf(x_init, y_init, z_init, l_init, m_init, n_init,op_init = op_init)

        if op_args.cal_opd==True:
            opd = self.from_last_surf2exit(x_out, y_out, z_out, l, m, n, opl, bfl=bfl[:, op_args.wave_idx:op_args.wave_idx + 1, :],efl=efl)
        else:
            opd = 0.

        # 像面信息处理：如果前面都没算梯度，这里自然不会有梯度
        spotsize, avg_y_list, color_loss = self.last_lens2_image(x_out, y_out, z_out, l, m, n,bfl)
        return spotsize, bfl, efl, avg_y_list, loss_intersect,loss_angle, color_loss, loss_curve_all,opd

    def plot_model_lens(self,op_args,data_file_name):
        # 首先计算th_for_ray_trace
        with torch.no_grad():
            th_for_ray_trace = self.cal_th_for_trace()
            efl, bfl = self.cal_efl_bfl(th_for_ray_trace)
            # 然后正向追迹一下，记录一些点
            N,surf_num = self.cv.shape
            # 这里需要记录一些特殊的点
            y_max_list = torch.zeros((N, surf_num + 1)).cuda()
            # 追迹前的准备
            x_out, y_out, z_out, l, m, n = x_init.data, y_init.data, z_init.data, l_init.data, m_init.data, n_init.data
            center = x_out.shape[-1]//2
            zero_indices = (torch.abs(x_out[0][0]) < 0.001)
            x_zero = x_out[0][0][zero_indices]
            # x_out, y_out, z_out, l, m, n = x_init , y_init , z_init , l_init , m_init , n_init
            y_zero = torch.zeros((N,surf_num + 2, x_zero.shape[0])).cuda()
            z_zero = torch.zeros((N,surf_num + 2, x_zero.shape[0])).cuda()
            y_zero[:,0] = y_init[0][0][zero_indices]
            z_zero[:,0] = z_init[0][0][zero_indices]
            for i in range(self.cv.shape[1]):
                x_out, y_out, z_out, l, m, n, _, _, _, _ = \
                    ray_trace_newton(x_out, y_out, z_out, l, m, n, self.decenter_x[:, :, i:i + 1],
                                     self.decenter_y[:, :, i:i + 1], th_for_ray_trace[:, i:i + 1].unsqueeze(-1),
                                     self.tilt_l[:, :, i:i + 1],
                                     self.tilt_m[:, :, i:i + 1], zero_1,
                                     self.k[:, i:i + 1].unsqueeze(-1), self.cv[:, i:i + 1].unsqueeze(-1),
                                     self.rn[:, :, i:i + 1],
                                     self.rn[:, :, i + 1:i + 2], self.omit[i], self.power_para[:, i:i + 1, :],
                                     self.th_label[i], cal_op=op_args.cal_opd, edge=lens_args.edge_min)
                y_zero[:,i + 1, :] = y_out[:,0,zero_indices]
                z_zero[:,i + 1, :] = z_out[:,0,zero_indices]
                y_max_list[:, i] = torch.max(torch.max(torch.abs(y_out), dim=-1).values, dim=-1).values
            x_out, y_out, z_out, l, m, n, op, _, _, _ = \
                ray_trace_newton(x_out, y_out, z_out, l, m, n, 0, 0, bfl[:, op_args.wave_idx:op_args.wave_idx + 1, :],
                                 zero_3_dim, zero_3_dim, zero_1, 0, 0., self.rn[:, :, -1:],self.rn[:, :, -1:], 1, 0, cal_op=False)
            spotsize, _,_,_,_ = cal_mlzi_spot_size(x_out, y_out, self.field_num)
            if op_args.N<100:
                print(spotsize)
            # print(spotsize)
            y_zero[:,-1, :] = y_out[:,0,zero_indices]
            z_zero[:,-1, :] = z_out[:,0,zero_indices]
            # 用来判断镜头最大高度
            y_max_list[:, -1] = torch.max(torch.max(torch.abs(y_out), dim=-1).values, dim=-1).values
            # 绘制点列图
            filename = op_args.file_name + '/' + data_file_name
            # plot_spot(x_out, y_out, lens_args.target_efl,lens_args.half_apture_max, field_y,self.field_num,filename,wave_num=3)

            th_for_plot = torch.cat((self.th, bfl[:, 0:1, 0]), dim=-1)
            y_max_list_all = y_max_list.data.cpu().numpy()
            cv = self.cv.data.cpu().numpy()
            th_for_plot = th_for_plot.data.cpu().numpy()
            k = self.k.data.cpu().numpy()
            power = self.power_para.data.cpu().numpy()
            z_zero =z_zero.data.cpu().numpy()
            y_zero = y_zero.data.cpu().numpy()

            plot_lens(y_max_list_all,th_for_plot,lens_args.half_apture_max,cv,k,power,y_zero,z_zero,filename,op_args.sample_delta)
            # 追迹部分准备完毕，开始使用特定的数据绘图

if __name__ == "__main__":
    N_list = [500,1000,1500,2000,2500]


    # Important experimental setups
    op_args_para = argparse.ArgumentParser(description='Transformer')
    op_args_para.add_argument('--file_name', type=str, default='Ultraviolet',help='System type')# cooke, Ultraviolet,6p
    op_args_para.add_argument('--method_name', type=str, default='GISG_GD',  help='Which algorithm to use') #GISG_GD, GD, CURR, PSO_GD,GIG_GD,PSO_GISG_GD
    op_args_para.add_argument('--optimiser', type=str, default='',   help='Gradient desent or DSLM: if not "LM", it be will adam')
    op_args_para.add_argument('--experiment_name', type=str, default='',   help='Experimental Notes, it will be added in the file name of output')
    op_args_para.add_argument('--epoch', type=int, default=10, help='')
    op_args_para.add_argument('--stop_time', type=int, default=100, help='T_stop_tim in thsis,if optimiser=LM, set stop_time = 10')
    op_args_para.add_argument('--N', type=int, default=20, help='Number of systems optimized at the same time')
    op_args_para.add_argument('--lr_max', type=float, default=0.01, help='base learning rate')

    op_args_para.add_argument('--sample_delta', type=int, default=7,   help='Number of light samples in the diaphragm')
    op_args_para.add_argument('--repeat_exp_tim', type=int, default=1, help='Number of repeated experiments')
    op_args = op_args_para.parse_args()
    if op_args.file_name=='6p':
        eff_v = 0.8;field_num = 6
    else:
        eff_v = 0.3;field_num = 4
    op_args_para.add_argument('--field_num', type=int, default=field_num, help='Number of fields during optimization')
    op_args_para.add_argument('--eff_v', type=float, default=eff_v,   help='system effectiveness threshold')
    op_args = op_args_para.parse_args()
    op_args_para.add_argument('--init_spot_constrain', type=float, default=1 * op_args.field_num,help='Limitations on the size of the dot plot when filtering starting points')
    op_args_para.add_argument('--stop_fix', type=bool, default=False,help='stop fix in specific interval or not')
    op_args_para.add_argument('--rand_guidence', type=bool, default=False,help='use rand information to guide')
    op_args = op_args_para.parse_args()
    if op_args.method_name == 'PSO_GD' or op_args.method_name == 'PSO_GISG_GD':
        Two_stage=True
        op_args.N = 8000
    else:
        Two_stage=False
    op_args_para.add_argument('--Two_stage', type=bool, default=Two_stage, help='Two stage strategy')
    op_args_para.add_argument('--Epoch_1', type=int, default=4000, help='first stage time')
    op_args_para.add_argument('--Epoch_2', type=int, default=4000, help='Second stage time')


    lr_adj = 100 # Learning rate decay cycle
    # Lens parameters are modified along here into the lens_para.py file
    lens_args, weight_arg = arg_select(op_args.file_name)

    # Secondary parameter settings
    with torch.no_grad():
        op_args_para.add_argument('--v_c_max', type=float, default=lens_args.cmax / 400.,help='The maximum value of the guided velocity of the curvature')
        op_args_para.add_argument('--v_th_max', type=float, default=(lens_args.th_a_max - lens_args.th_a_min) / 400,help='The maximum value of the guided velocity of the thickness')
        op_args_para.add_argument('--v_rn_max', type=float, default=0.01,help='The maximum value of the guided velocity of the rn')
        op_args_para.add_argument('--v_k_max', type=float, default=0.01,help='The maximum value of the guided velocity of the K')
        op_args_para.add_argument('--k_as_max', type=float, default=100, help='The maximum value of K')
        op_args_para.add_argument('--k_as_min', type=float, default=-100, help='The minimal value of K')
        op_args_para.add_argument('--power_max_value', type=float, default=0.5, help='The minimal value of power')
        op_args = op_args_para.parse_args()
        if op_args.method_name=='GISG_GD':
            weight_guidence = 1; weight_grd=1
        elif op_args.method_name=='GD':
            weight_guidence = 0;weight_grd = 1
        elif op_args.method_name=='GIG_GD':
            weight_guidence = 1;weight_grd = 1
        elif op_args.method_name=='CURR':
            weight_guidence = 0;weight_grd = 1
        elif op_args.method_name == 'PSO_GD' or op_args.method_name == 'PSO_GISG_GD':
            weight_guidence = 1;weight_grd = 0

        # Adjusting the learning rate for each parameter from the base learning rate
        # If the method is PSO, all lr will be set zero, no worry
        if op_args.file_name == '6p':
            lc=1; lth=0.5;lk = 1;lrpower = 0.0001
        if op_args.file_name == 'cooke':
            lc=1; lth=1;lk = 0;lrpower = 0.0000
        if op_args.file_name == 'Ultraviolet':
            lc=0.05; lth=5;lk = 0;lrpower = 0.0000
        else:
            lc=1; lth=1;lk = 0;lrpower = 0.0000
        lr_c, lr_th, lr_k, lr_power, lr_rn = lr_set(lc, lth, lk, lrpower, 0.0, op_args.lr_max,weight_grd)  # cooke:11,UV:0.05,5#6p 1  0.5

        # weight_guidence decide to enable or disable the guidence message,We recommend using 1

        op_args_para.add_argument('--weight_guidence', type=float, default=weight_guidence, help='Global information weights')
        op_args_para.add_argument('--weight_grd', type=float, default=weight_grd, help='')
        op_args = op_args_para.parse_args()
        # weight_local Controls the local optimization speed of the PSO
        if op_args.weight_grd > 0:
            weight_local = 0  # Local optimization weights
        else:
            weight_local = 2
        # weight_global Controlling the intensity of the guidence message, usually not as useful as speed max
        weight_global = 2  # global optimization weights

    # used to test lens sensitivity, ray aiming, change dominant wavelengths
    with torch.no_grad():
        op_args_para.add_argument('--wave_idx', type=int, default=1, help='which color is main wavelength')
        op_args_para.add_argument('--rn_vara', type=bool, default=False,   help='Whether the material is variable')
        op_args_para.add_argument('--cal_opd', type=bool, default=False,   help='Whether calculate opd')
        op_args_para.add_argument('--offer_init', type=bool, default=False,   help='Whether offer init systems')
        op_args_para.add_argument('--ray_aming', type=bool, default=False,  help='Whether use paraxial ray aming')
        op_args_para.add_argument('--desensitive', type=bool, default=False,    help='Whether desensitization training')
        op_args_para.add_argument('--data_idx', type=int, default=8,  help='When reading the system from excel, select which system')
        op_args_para.add_argument('--k_if_init', type=bool, default=False, help='Whether K is randomly initialized')
        op_args = op_args_para.parse_args()

        #



    for exe_time in range(op_args.repeat_exp_tim):

        op_args.experiment_name = op_args.method_name+'_'+op_args.optimiser+'_' +str(exe_time)
        if op_args.rand_guidence == True:
            op_args.experiment_name = op_args.experiment_name +'rand'
        if op_args.Two_stage==True:
            op_args.experiment_name = op_args.experiment_name + 'Two_stage'
        # Some parameter initialization procedures, which usually do not need to be modified
        with torch.no_grad():

            w = 1

            # 初始的系统参数
            # lens_args.power_omit = np.ones(lens_args.surf_num)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            # if stop is fix:
            th_label = generate_th_label(lens_args.surf_num, lens_args.stop_idx)
            # else:
            if op_args.stop_fix ==False:
                th_label = generate_th_label(lens_args.surf_num, 1)
            # lens_args.k_label = np.zeros(lens_args.surf_num)
            # lens_args.power_omit = [0,0,0,0,0,0,0,0,0,0,0,0,1,1]
            field_weight = np.linspace(0, 1, op_args.field_num)
            weight_spot_size = (1 + 0 * torch.from_numpy(field_weight).cuda())
            weight_spot_size = weight_spot_size / weight_spot_size.sum()*weight_arg.weight_spot
            v_c,v_th,v_rn,v_k = v_max_min(op_args.N, lens_args.surf_num, op_args.v_c_max, lens_args.length_num, op_args.v_th_max,op_args.v_rn_max, op_args.weight_grd,op_args.v_k_max)
            th_min,th_max,k_min,k_max,power_max,rn_min_list,rn_max_list = value_max_min(lens_args.length_num,th_label,lens_args.th_g_min,lens_args.th_g_max,lens_args.th_a_min,lens_args.th_a_max,
                                                                lens_args.th_stop_min,lens_args.th_stop_max,lens_args.k_label,op_args.k_as_min,op_args.k_as_max,lens_args.power_num,lens_args.surf_num,op_args.power_max_value,lens_args.rn_min,lens_args.rn_max)

            field_largest = lens_args.field_max
            field_y = (np.linspace(0, lens_args.field_max, op_args.field_num) ).tolist()
            field_x = None
            y_field_ideal,x_field_ideal,op_args.field_num = generate_field_and_ideal(field_y,lens_args.target_efl,0,field_x=field_x)
            x_init, y_init, z_init, l_init, m_init, n_init, number_of_gridpoints, op_args.sample_delta,op_init = grid_data_of_staring_point_para(
                lens_args.half_apture_max, field_y,field_x= field_x,
                sample_delta=op_args.sample_delta)
            if op_args.offer_init==True:
                idx = 0
                with open('../data/datatxt/6p/power/6p_power_all_sgd_pso_' + str(idx) + '.pkl', 'rb') as f:
                    power_para = pickle.load(f)
                power_temp = torch.from_numpy(power_para).cuda()
                data = torch.from_numpy(np.loadtxt('../data/datatxt/6p/6p_all_sgd_pso_' + str(idx) + '.txt')).cuda().transpose(
                    0, 1)

                mask = torch.sort(data[:, -1]).indices[:op_args.N]
                data = data[mask]
                c = 1 / data[:, 14:28].cuda()
                th = data[:, :14].cuda()
                k = data[:, 28:-1].cuda()
                power_temp = power_temp[mask, :, :]

        # test_code()
        # plot_lens_code('cooke_all_EA_GD_adam_0',old_data=False) # cooke_all_sgd_pso_0,cooke_all_EA_GD_adam_0
        # generate eff start points
        with torch.no_grad():
            seed_torch(23 + exe_time)
            c,th,rn_model,loss,c_best,c_person_best,th_best,th_person_best,rn_best,rn_person_best,loss_person_best,loss_best_list,loss_best_all,loss_list,loss_last_one,person_best_i,\
            judge_guidence,k,k_person_best,k_best,power_temp,power_person_best=find_init()
            c_best_all = c_best
            th_best_all = th_best
            rn_best_all = rn_best
            if lr_k!=0:
                k_best_all = k_best
            else:
                k_best_all = 0.
            if lr_power!=0:
                power_best_all = power_temp[0:1]
            else:
                power_best_all = 0.



        # 设置基本模型、选择优化器等：
        with torch.enable_grad():
            model = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, th_label,
                                              omit=lens_args.power_omit, field_num=op_args.field_num, stop_idx=lens_args.stop_idx,wave_number=lens_args.wave_number).cuda()

            # 使用找到的起始点

            model_test = spot_size_layer_multiwave(lens_args.surf_num, lens_args.length_num, lens_args.power_num, th_label,
                                                   omit=lens_args.power_omit, field_num=op_args.field_num, stop_idx=lens_args.stop_idx,wave_number=lens_args.wave_number).cuda()
            if op_args.optimiser=='LM':
                if op_args.file_name == '6p':
                    params_dict = [model.cv, model.th, model.k, model.power_para]
                else:
                    params_dict = [model.cv, model.th]
                optimizer = optim.LBFGS(params_dict,op_args.lr_max,max_iter=10)
            else:
                params_dict = [
                    {'params': model.cv, 'lr': lr_c},{'params': model.th, 'lr': lr_th},{'params': model.k, 'lr': lr_k},
                    {'params': model.power_para, 'lr': lr_power}, {'params': model.rn, 'lr': lr_rn}
                ]
                optimizer = optim.Adam(params_dict,weight_decay = 0,betas=(0.9,0.99))



            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=lr_adj, eta_min=0)

            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(op_args.epoch/30), gamma=0.8)
            if op_args.rn_vara==True:
                model.rn.data = rn_model
                rn_temp, material_idx_last = find_best_material(rn_database, model.rn.data, rn_min_list,
                                                                      rn_max_list)
            loss_temp=0.#torch.ones((op_args.N,1)).cuda() * 1e8
            model.cv.data = c
            model.th.data = th
            if lr_k != 0:
                model.k.data = k
            if lr_power != 0:
                model.power_para.data = power_temp
            if op_args.rn_vara == True:
                model.rn.data = rn_temp



        # 这个参数用来调整生成数据集的指标数量


        if op_args.Two_stage == True:
            op_args.weight_grd = 0
            op_args.weight_guidence = 1
        # specification_sample = 1
        # for sp_sample in range(specification_sample):
        #     # field_now = lens_args.field_min + (lens_args.field_max - lens_args.field_min) /(specification_sample-1) * sp_sample
        #     # half_apture_now = lens_args.half_apture_min + (lens_args.half_apture_max - lens_args.half_apture_min) /(specification_sample-1) * sp_sample
        #     # print('field: ', field_now, 'half_ap: ', half_apture_now)
        #     # field = (np.linspace(0, 1, op_args.field_num) * field_now).tolist()
        #     # x_init, y_init, z_init, l_init, m_init, n_init, number_of_gridpoints, op_args.sample_delta, op_init = grid_data_of_staring_point_para(
        #     #     half_apture_now, field,
        #     #     op_args.sample_delta=op_args.sample_delta)
        #     # y_field_ideal = lens_args.target_efl * torch.tan(torch.tensor(field) / 180 * math.pi)
        #     # y_field_ideal = y_field_ideal.unsqueeze(0).cuda()
        #     # if sp_sample!=0:
        #     #     loss_person_best = loss_person_best + 10e8
        #     #     loss_best_all = 10e8
        #     #     op_args.epoch = 500
        #     # op_args.eff_v = 0.3 + (0.8-0.3)/(specification_sample-1) * sp_sample
        #     # op_args.experiment_name = op_args.method_name + '_' + str(exe_time) + '_'+str(sp_sample)
        loss_sum = []
        record_stage = 0
        stage_judge = op_args.Epoch_1
        for i in range(op_args.epoch):

                # 如果是两阶段算法，则需要全局算法和局部算法的权重进行调整
                if op_args.Two_stage == True:
                    record_stage += 1
                    if record_stage % stage_judge==0 and stage_judge==op_args.Epoch_1 :
                        op_args.weight_grd = 1
                        op_args.weight_guidence =1
                        stage_judge = op_args.Epoch_2
                        record_stage = 1
                        op_args.stop_time = 100
                        person_best_i = person_best_i *0 # 进入新阶段，将优化停止计数先回零
                    # 判断进入一阶段
                    if record_stage % stage_judge==0 and stage_judge==op_args.Epoch_2:
                        op_args.weight_grd = 0
                        op_args.weight_guidence =1
                        stage_judge = op_args.Epoch_1
                        record_stage = 1
                    # weight_local Controls the local optimization speed of the PSO
                    if op_args.weight_grd > 0:
                        weight_local = 0  # Local optimization weights
                    else:
                        weight_local = 2
                    # weight_global Controlling the intensity of the guidence message, usually not as useful as speed max
                    weight_global = 2  # global optimization weights
                # if i==op_args.epoch/2:
                #     weight_arg.weight_spot = 0
                #     weight_arg.weight_opd = 0.1
                #     op_args.cal_opd = True
                #     loss_person_best = loss_person_best + 10e8
                #     loss_best_all = 10e8
                #     optimizer = optim.Adam(params_dict, weight_decay=0, betas=(0.9, 0.99))
                #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=lr_adj,
                #                                                                      eta_min=0)
                #
                #     scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8)
                if op_args.method_name=='CURR':
                    field_now = lens_args.field_min + (lens_args.field_max - lens_args.field_min) * abs(
                        math.sin((i ) / 2 / (op_args.epoch ) * math.pi))  # 15
                    half_apture_now = lens_args.half_apture_min + (lens_args.half_apture_max - lens_args.half_apture_min) * math.sin(
                        (i ) / 2 / (op_args.epoch ) * math.pi)
                    print('field: ', field_now, 'half_ap: ', half_apture_now)
                    field = (np.linspace(0, 1, op_args.field_num) * field_now).tolist()
                    x_init, y_init, z_init, l_init, m_init, n_init, number_of_gridpoints, op_args.sample_delta,op_init = grid_data_of_staring_point_para(
                        half_apture_now, field,
                        sample_delta=op_args.sample_delta)
                    y_field_ideal = lens_args.target_efl * torch.tan(torch.tensor(field) / 180 * math.pi)
                    y_field_ideal = y_field_ideal.unsqueeze(0).cuda()
                start = time.time()

                # 全局算法是EA还有PSO有不同的参数调整策略
                if i != 0:
                    model, v_c, v_th, rn_temp,v_rn,loss_last_one,v_k = pos_one_time_iter(model,v_c,v_th,v_rn,i,loss_last_one,loss_temp,v_k=v_k)


                # aspherical lens need setting last two surfaces to zero



                # if rn is variable, rn need to use the nearest lens in lens database

                if op_args.rn_vara==True:
                    rn_temp,material_idx = find_best_material(rn_database, model.rn.data,rn_min_list,rn_max_list)
                    rn_change = torch.sum(torch.abs(material_idx - material_idx_last),dim=-1,keepdim=True)!=0
                    print('rn_change num: ',torch.sum(rn_change).item())
                    material_idx_last = material_idx
                    model.rn.data = rn_temp


                # GD and LM have different code:
                if op_args.optimiser == 'LM':
                    with torch.no_grad():
                        loss_temp, loss_back, spot_size, dist_loss, color_loss, efl, bfl ,TOTR,f,opd_loss= cal_loss(model)
                    def closure():
                        global optimizer
                        optimizer.zero_grad()
                        loss_temp, loss_back, spot_size, dist_loss, color_loss, efl, bfl, TOTR, f, opd_loss = cal_loss(
                            model)
                        mask_judge_nan = (loss_temp == 1e3).squeeze(-1)
                        model.cv.data[mask_judge_nan] = c_person_best[mask_judge_nan]
                        model.th.data[mask_judge_nan] = th_person_best[mask_judge_nan]
                        if lr_k != 0:
                            model.k.data[mask_judge_nan] = k_person_best[mask_judge_nan]
                        if lr_power != 0:
                            model.power_para.data[mask_judge_nan] = power_person_best[mask_judge_nan]

                        loss_back = loss_back.sum()
                        # print('loss back sum ',loss_back)
                        loss_back.backward()
                        optimizer = grad_adjust(model, optimizer)
                        if loss_temp.min()==1000:
                            op_args.lr_max = op_args.lr_max*0.8
                            optimizer = optim.LBFGS(params_dict, op_args.lr_max)
                        return loss_temp.sum()
                else:
                    loss_temp, loss_back, spot_size, dist_loss, color_loss, efl, bfl, TOTR, f, opd_loss = cal_loss(
                        model)

                # record period data and update best record
                c_person_best, c_best, c_best_all, th_person_best, th_best, th_best_all, rn_person_best, rn_best, rn_best_all, idx_best, \
                person_best_i, loss_person_best, loss_person_best_median, loss_best_all, judge_guidence,\
                k_person_best,k_best,k_best_all,power_person_best,power_best_all,power_best= \
                    find_and_save_best(loss_person_best, loss_best_all, c_person_best, th_person_best,
                                       rn_person_best, person_best_i, c_best_all, th_best_all, rn_best_all,
                                       k_person_best, k_best_all,power_person_best,power_best_all)

                # show all systems' total loss,
                loss_person_best_sum = loss_person_best.sum()
                loss_sum.append(loss_person_best_sum.item())
                print('loss_person_best sum: ',loss_person_best_sum )

                # if do not use local descent
                if op_args.weight_grd!=0:
                    if op_args.optimiser == 'LM':
                        loss_back= optimizer.step(closure)
                    else:
                        optimizer.zero_grad()
                        loss_back.sum().backward()
                        optimizer = grad_adjust(model, optimizer)
                        optimizer.step()

                    scheduler.step()
                    try:
                        scheduler2.step()
                    except:
                        pass


                # replace error sys with their own best history
                with torch.no_grad():
                    mask_judge_nan = (loss_temp==1e3).squeeze(-1)
                    model.cv.data[mask_judge_nan] = c_person_best[mask_judge_nan]
                    model.th.data[mask_judge_nan] = th_person_best[mask_judge_nan]
                    if lr_k!=0:
                        model.k.data[mask_judge_nan] = k_person_best[mask_judge_nan]
                    if lr_power!=0:
                        model.power_para.data[mask_judge_nan] = power_person_best[mask_judge_nan]
                    if op_args.rn_vara == True:
                        model.rn.data[mask_judge_nan] = rn_person_best[mask_judge_nan]
                    # 将所有结果替换成目前最好的结果
                    if op_args.rn_vara==True:
                        best_glass_name = []
                        for glass in range(material_idx.shape[-1]):
                            best_glass_name.append(glass_name[int(material_idx[idx_best][0,glass].item())])




                # print some temp results to show the process
                with torch.no_grad():
                    end = time.time()
                    loss_best_list[i] = (loss_temp[idx_best])[0][0]
                    print('exe_time:',exe_time,'.epoch: ', i, 'time: ', end - start, 'efl', efl[idx_best][0][op_args.wave_idx].item(), 'TOTR',
                          TOTR[idx_best].item(),'bfl',bfl[idx_best][0][op_args.wave_idx].item() )
                    print('loss_best: ', (loss_temp[idx_best] ).item(), 'distort: ', dist_loss[idx_best].item())
                    if weight_arg.weight_color!=0:
                        print( 'color: ',color_loss[idx_best].item())
                    if weight_arg.weight_opd!=0:
                        print('opd loss:', opd_loss[idx_best].item())
                    if op_args.rn_vara==True:
                        print('glass ',best_glass_name)
                    # if (loss_temp[idx_best] ).item()<0.8:
                    #     op_args.desensitive =True
                    #     loss_person_best = loss_person_best + 1e8
                    #     loss_best_all = 1e8

                    print('spot_size:', spot_size[idx_best] )

                    # print('glass ',best_glass_name)





                    # 保存结果
                    loss_list.append((loss_temp[idx_best][0][0]).item())
                    # 保存结果

                # save results
                if i + 1 == op_args.epoch:
                    save_lens_opt_result(loss_list, loss_sum,c_best, th_best, rn_person_best, rn_best_all,
                                         c_best_all, th_best_all,k_best,k_best_all,power_best,power_best_all)


