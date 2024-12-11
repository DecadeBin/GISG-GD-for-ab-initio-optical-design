
from DFRT import *
import argparse
import torch


def find_rn_data_from_glass_base(lens_material,materialbase,rn_database):
    indices = [materialbase.index(item) for item in lens_material]
    rn_glass = rn_database[indices]
    rn = torch.ones((2*len(lens_material)+2),3)
    for i in range(len(lens_material)):
        rn[2*i+1,:] = rn_glass[i]
    rn = rn.unsqueeze(0).permute(0,2,1)
    return rn


# cooke:-----------------------------------------------------------------------------------
with torch.no_grad():
    rn_cooke = torch.tensor(  [[[1, 1.6275563488, 1, 1.6284791544, 1, 1.6275563488, 1, 1],
                                [1, 1.6204099651, 1, 1.6165918071, 1, 1.6204099651, 1, 1],
                                [1, 1.6172716599, 1, 1.6116452953, 1, 1.6172716599, 1, 1]]])
    cooke_lens_rescale = 1
    cooke_para = argparse.ArgumentParser(description='Transformer')
    cooke_para.add_argument('--target_efl', type=float, default=50*cooke_lens_rescale)
    cooke_para.add_argument('--rn_list', type=float, default=rn_cooke)
    cooke_para.add_argument('--field_max', type=float, default=15)
    cooke_para.add_argument('--field_min', type=float, default=15/2)
    cooke_para.add_argument('--half_apture_max', type=float, default=8/2)
    cooke_para.add_argument('--half_apture_min', type=float, default=4/2)
    cooke_para.add_argument('--cmax', type=float, default=1/5.)
    cooke_para.add_argument('--c_init_max', type=float, default=1 / 10)
    cooke_para.add_argument('--k_init_max', type=float, default=0)
    cooke_para.add_argument('--wave_number', type=int, default=rn_cooke.shape[1])
    cooke_para.add_argument('--surf_num', type=int, default=6)
    cooke_para.add_argument('--length_num', type=int, default=6)
    cooke_para.add_argument('--th_g_min', type=float, default=2*cooke_lens_rescale)
    cooke_para.add_argument('--th_g_max', type=float, default=6*cooke_lens_rescale)
    cooke_para.add_argument('--th_a_min', type=float, default=1*cooke_lens_rescale)
    cooke_para.add_argument('--th_a_max', type=float, default=10*cooke_lens_rescale)
    # if stop is float, th_stop_min should be negative
    cooke_para.add_argument('--th_stop_min', type=float, default=-30*cooke_lens_rescale)
    cooke_para.add_argument('--th_stop_max', type=float, default=2 * cooke_lens_rescale)
    cooke_para.add_argument('--bfl_max', type=float, default=100*cooke_lens_rescale)
    cooke_para.add_argument('--bfl_min', type=float, default=20*cooke_lens_rescale)
    cooke_para.add_argument('--edge_min', type=list, default=[0.1,0.2,-30])# 边缘厚度，[空气，玻璃，光阑]
    cooke_para.add_argument('--TOTR_max', type=float, default=500)
    cooke_para.add_argument('--rn_max', type=float, default=2)
    cooke_para.add_argument('--rn_min', type=float, default=1.3)
    cooke_para.add_argument('--power_num', type=int, default=1)
    cooke_para.add_argument('--stop_idx', type=float, default=4)
    cooke_para.add_argument('--dist_max', type=float, default=0.01)
    cooke_para.add_argument('--color_max', type=float, default=0.008)
    cooke_args = cooke_para.parse_args()
    cooke_para.add_argument('--k_label', type=float, default=[0 for _ in range(cooke_args.surf_num)])
    cooke_para.add_argument('--power_omit', type=float, default=[1 for _ in range(cooke_args.surf_num)])
    cooke_para.add_argument('--decenter_max', type=float, default=0.008)
    cooke_para.add_argument('--decenter_z_max', type=float, default=0.025)
    cooke_para.add_argument('--tilt_max', type=float, default=0.8/60 / 180 *math.pi)
    cooke_args = cooke_para.parse_args()

    weight_arg_cooke = argparse.ArgumentParser(description='Transformer')
    weight_arg_cooke.add_argument('--weight_efl', type=float, default=1)
    weight_arg_cooke.add_argument('--weight_TOTR', type=float, default=1,help='weight of total length')
    weight_arg_cooke.add_argument('--weight_angle', type=float, default=0.001,help='weight of Light transmission angle')
    weight_arg_cooke.add_argument('--weight_spot', type=float, default=50,help='weight of spot size')
    weight_arg_cooke.add_argument('--weight_opd', type=float, default=0,help='weight of optical length difference')
    weight_arg_cooke.add_argument('--weight_spot_std', type=float, default=0)
    weight_arg_cooke.add_argument('--weight_c_board', type=float, default=1.,help='weight of curve constrains')
    weight_arg_cooke.add_argument('--weight_th_board', type=float, default=1.,help='weight of thicknesses constrains')
    weight_arg_cooke.add_argument('--weight_curve', type=float, default=0.1,help='Control the angle between surface normal and optical axis')
    weight_arg_cooke.add_argument('--weight_c_std', type=float, default=0.00,help='weight to control the standard deviation of curvature of surfaces')
    weight_arg_cooke.add_argument('--weight_color', type=float, default=10,help='weight to control axial aberration')
    weight_arg_cooke.add_argument('--weight_intersect', type=float, default=1,help='weight to control the mininal distance between surfaces')
    weight_arg_cooke.add_argument('--weight_dist', type=float, default=1,help='weight of distortion')
    weight_arg_cooke.add_argument('--weight_bfl', type=float, default=1,help='weihgt of back focal length')
    weight_arg_cooke = weight_arg_cooke.parse_args()


# UV:-----------------------------------------------------------------------------------
with torch.no_grad():
    rn_UV = torch.tensor(  [[[1, 1.4583622581, 1, 1.4941636601, 1, 1.4583622581, 1, 1.4583622581, 1, 1.4941636601, 1,1.4583622581,1,1],
                            [1, 1.4418536698, 1, 1.4701161181, 1, 1.4418536698, 1, 1.4418536698, 1, 1.4701161181, 1,1.4418536698,1,1],
                        [1, 1.4357592613, 1, 1.4612804078, 1, 1.4357592613, 1, 1.4357592613, 1, 1.4612804078, 1,1.4357592613,1,1]
                       ]])
    UV_para = argparse.ArgumentParser(description='Transformer')
    UV_para.add_argument('--target_efl', type=float, default=50)
    UV_para.add_argument('--rn_list', type=float, default=rn_UV)
    UV_para.add_argument('--field_max', type=float, default=7)
    UV_para.add_argument('--field_min', type=float, default=7/2)
    UV_para.add_argument('--half_apture_max', type=float, default=33.33/2/2)
    UV_para.add_argument('--half_apture_min', type=float, default=33.33/2/2/2)
    UV_para.add_argument('--cmax', type=float, default=1/10.)
    UV_para.add_argument('--c_init_max', type=float, default=1 / 20)
    UV_para.add_argument('--k_init_max', type=float, default=0)
    UV_para.add_argument('--wave_number', type=int, default=rn_UV.shape[1])
    UV_para.add_argument('--surf_num', type=int, default=12)
    UV_para.add_argument('--length_num', type=int, default=12)
    UV_para.add_argument('--th_g_min', type=float, default=3.5)
    UV_para.add_argument('--th_g_max', type=float, default=10)
    UV_para.add_argument('--th_a_min', type=float, default=1)
    UV_para.add_argument('--th_a_max', type=float, default=10)
    UV_para.add_argument('--th_stop_min', type=float, default=10)
    UV_para.add_argument('--th_stop_max', type=float, default=15)
    UV_para.add_argument('--bfl_max', type=float, default=100)
    UV_para.add_argument('--bfl_min', type=float, default=10)
    UV_para.add_argument('--edge_min', type=list, default=[0.1,0.2,10])# 边缘厚度，[空气，玻璃，光阑]
    UV_para.add_argument('--TOTR_max', type=float, default=1000)
    UV_para.add_argument('--rn_max', type=float, default=2)
    UV_para.add_argument('--rn_min', type=float, default=1.3)
    UV_para.add_argument('--power_num', type=int, default=1)
    UV_para.add_argument('--stop_idx', type=float, default=1)
    UV_para.add_argument('--dist_max', type=float, default=0.05)
    UV_para.add_argument('--color_max', type=float, default=0.008)
    UV_args = UV_para.parse_args()
    UV_para.add_argument('--k_label', type=float, default=[0 for _ in range(UV_args.surf_num)])
    UV_para.add_argument('--power_omit', type=float, default=[1 for _ in range(UV_args.surf_num)])
    UV_para.add_argument('--decenter_max', type=float, default=0.0)
    UV_para.add_argument('--decenter_z_max', type=float, default=0.0)
    UV_para.add_argument('--tilt_max', type=float, default=0/60 / 180 *math.pi)
    UV_args = UV_para.parse_args()


    weight_arg_UV = argparse.ArgumentParser(description='Transformer')
    weight_arg_UV.add_argument('--weight_efl', type=float, default=1)
    weight_arg_UV.add_argument('--weight_TOTR', type=float, default=1,help='weight of total length')
    weight_arg_UV.add_argument('--weight_angle', type=float, default=0.001,help='weight of Light transmission angle')
    weight_arg_UV.add_argument('--weight_spot', type=float, default=50,help='weight of spot size')
    weight_arg_UV.add_argument('--weight_opd', type=float, default=0,help='weight of optical length difference')
    weight_arg_UV.add_argument('--weight_spot_std', type=float, default=0)
    weight_arg_UV.add_argument('--weight_c_board', type=float, default=1.,help='weight of curve constrains')
    weight_arg_UV.add_argument('--weight_th_board', type=float, default=1.,help='weight of thicknesses constrains')
    weight_arg_UV.add_argument('--weight_curve', type=float, default=0.1,help='Control the angle between surface normal and optical axis')
    weight_arg_UV.add_argument('--weight_c_std', type=float, default=0.00,help='weight to control the standard deviation of curvature of surfaces')
    weight_arg_UV.add_argument('--weight_color', type=float, default=10,help='weight to control axial aberration')
    weight_arg_UV.add_argument('--weight_intersect', type=float, default=1,help='weight to control the mininal distance between surfaces')
    weight_arg_UV.add_argument('--weight_dist', type=float, default=1,help='weight of distortion')
    weight_arg_UV.add_argument('--weight_bfl', type=float, default=1,help='weihgt of back focal length')
    weight_arg_UV = weight_arg_UV.parse_args()


# p6:-----------------------------------------------------------------------------------
with torch.no_grad():
    rn_p6 = torch.tensor(  [[[1, 1.5507793111, 1, 1.6529801712, 1, 1.6529801712, 1, 1.5507793111, 1, 1.5657991188, 1,1.6310676252,1,1.5225746862,1, 1],
                            [1, 1.5440019294, 1, 1.6340051301, 1, 1.6340051301, 1, 1.5440019294, 1, 1.5550029782, 1,1.6140046266,1,1.5170016092,1, 1],
                            [1, 1.5410515500, 1, 1.6263525314, 1, 1.6263525314, 1, 1.5410515500, 1, 1.5505161106, 1,1.6070931514,1,1.5145249284,1, 1]
                               ]])

    p6_para = argparse.ArgumentParser(description='Transformer')
    p6_para.add_argument('--target_efl', type=float, default=3.14)
    p6_para.add_argument('--rn_list', type=float, default=rn_p6)
    p6_para.add_argument('--field_max', type=float, default=42.5)
    p6_para.add_argument('--field_min', type=float, default=42.5/2)
    p6_para.add_argument('--half_apture_max', type=float, default=1.57/2) # 1.4272
    p6_para.add_argument('--half_apture_min', type=float, default=1.57/2/2)
    p6_para.add_argument('--cmax', type=float, default=1/1.)
    p6_para.add_argument('--c_init_max', type=float, default=1 /5)
    p6_para.add_argument('--k_init_max', type=float, default=0)
    p6_para.add_argument('--wave_number', type=float, default=rn_p6.shape[1])
    p6_para.add_argument('--surf_num', type=int, default=14)
    p6_para.add_argument('--length_num', type=int, default=14)
    p6_para.add_argument('--th_g_min', type=float, default=0.25)
    p6_para.add_argument('--th_g_max', type=float, default=1.2)
    p6_para.add_argument('--th_a_min', type=float, default=0.02)
    p6_para.add_argument('--th_a_max', type=float, default=0.8)
    p6_para.add_argument('--th_stop_min', type=float, default=-4)
    p6_para.add_argument('--th_stop_max', type=float, default=0.2)
    p6_para.add_argument('--bfl_max', type=float, default=1.5)
    p6_para.add_argument('--bfl_min', type=float, default=0.3)
    p6_para.add_argument('--edge_min', type=list, default=[0.01,0.02,-4])# 边缘厚度，[空气，玻璃，光阑]
    p6_para.add_argument('--TOTR_max', type=float, default=7)
    p6_para.add_argument('--rn_max', type=float, default=2)
    p6_para.add_argument('--rn_min', type=float, default=1.3)
    p6_para.add_argument('--power_num', type=int, default=5)
    p6_para.add_argument('--stop_idx', type=float, default=1)
    p6_para.add_argument('--dist_max', type=float, default=0.3)
    p6_para.add_argument('--color_max', type=float, default=0.008)
    p6_para.add_argument('--decenter_max', type=float, default=0.000)
    p6_para.add_argument('--tilt_max', type=float, default=0)
    p6_para.add_argument('--k_label', type=float, default=[1,1,1,1,1,1,1,1,1,1,1,1,0,0])
    p6_para.add_argument('--power_omit', type=float, default=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    p6_args = p6_para.parse_args()

    weight_arg_6p = argparse.ArgumentParser(description='Transformer')
    weight_arg_6p.add_argument('--weight_efl', type=float, default=1)
    weight_arg_6p.add_argument('--weight_TOTR', type=float, default=1,help='weight of total length')
    weight_arg_6p.add_argument('--weight_angle', type=float, default=0.01,help='weight of Light transmission angle')
    weight_arg_6p.add_argument('--weight_spot', type=float, default=50,help='weight of spot size')
    weight_arg_6p.add_argument('--weight_opd', type=float, default=0,help='weight of optical length difference')
    weight_arg_6p.add_argument('--weight_spot_std', type=float, default=0)
    weight_arg_6p.add_argument('--weight_c_board', type=float, default=1.,help='weight of curve constrains')
    weight_arg_6p.add_argument('--weight_th_board', type=float, default=1.,help='weight of thicknesses constrains')
    weight_arg_6p.add_argument('--weight_curve', type=float, default=0.1,help='Control the angle between surface normal and optical axis')
    weight_arg_6p.add_argument('--weight_c_std', type=float, default=0.001,help='weight to control the standard deviation of curvature of surfaces')
    weight_arg_6p.add_argument('--weight_color', type=float, default=5,help='weight to control axial aberration')
    weight_arg_6p.add_argument('--weight_intersect', type=float, default=1,help='weight to control the mininal distance between surfaces')
    weight_arg_6p.add_argument('--weight_dist', type=float, default=1,help='weight of distortion')
    weight_arg_6p.add_argument('--weight_bfl', type=float, default=1,help='weihgt of back focal length')
    weight_arg_6p = weight_arg_6p.parse_args()
# def generate_middle_value(lens_args,weight_args,op_args):
#     # Generate some values that will be needed during the running of the algorithm

def arg_select(file_name):
    if file_name == 'cooke':
        args = cooke_args
        weight_args =weight_arg_cooke
    if file_name == 'Ultraviolet':
        args = UV_args
        weight_args = weight_arg_UV
    if file_name == '6p':
        args = p6_args
        weight_args = weight_arg_6p

    return args,weight_args

def generate_middle_value(op_args):
    lens_args, weight_arg = arg_select(op_args.file_name)














