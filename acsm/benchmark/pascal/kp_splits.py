import pdb

def get_kp_splits(kp_names, pascal_class):
    select_kp_ids = {}
    if pascal_class == 'horse' or pascal_class == 'cow' or pascal_class == 'sheep' :
        kpname2kpx = {kpname: kpx for kpx, kpname in enumerate(kp_names)}
        select_kp_ids['leg'] = []
        select_kp_ids['leg'].append(kpname2kpx['L_B_Elbow'],)
        select_kp_ids['leg'].append(kpname2kpx['L_B_Paw'],)
        select_kp_ids['leg'].append(kpname2kpx['L_F_Elbow'],)
        select_kp_ids['leg'].append(kpname2kpx['L_F_Paw'],)
        select_kp_ids['leg'].append(kpname2kpx['R_B_Elbow'],)
        select_kp_ids['leg'].append(kpname2kpx['R_B_Paw'],)
        select_kp_ids['leg'].append(kpname2kpx['R_F_Elbow'],)
        select_kp_ids['leg'].append(kpname2kpx['R_F_Paw'],)
        
        select_kp_ids['head'] = []
        select_kp_ids['head'].append(kpname2kpx['Nose'])
        select_kp_ids['head'].append(kpname2kpx['L_EarBase'])
        select_kp_ids['head'].append(kpname2kpx['L_Eye'])
        select_kp_ids['head'].append(kpname2kpx['R_Eye'])
        select_kp_ids['head'].append(kpname2kpx['R_EarBase'])
        
        select_kp_ids['torso'] = []
        select_kp_ids['torso'].append(kpname2kpx['Withers'])
        select_kp_ids['torso'].append(kpname2kpx['Throat'])
        select_kp_ids['torso'].append(kpname2kpx['TailBase'])

    if pascal_class == 'bird' :
        kpname2kpx = {kpname: kpx for kpx, kpname in enumerate(kp_names)}
        select_kp_ids['head'] = []
        select_kp_ids['head'].append(kpname2kpx['FHead'])
        select_kp_ids['head'].append(kpname2kpx['Crown'])
        select_kp_ids['head'].append(kpname2kpx['LEye'])
        select_kp_ids['head'].append(kpname2kpx['REye'])
        select_kp_ids['head'].append(kpname2kpx['Throat'])
        select_kp_ids['head'].append(kpname2kpx['Beak'])
        select_kp_ids['head'].append(kpname2kpx['Nape'])
        
        select_kp_ids['torso'] = []
        select_kp_ids['torso'].append(kpname2kpx['Belly'])
        select_kp_ids['torso'].append(kpname2kpx['Breast'])
        select_kp_ids['torso'].append(kpname2kpx['LWing'])
        select_kp_ids['torso'].append(kpname2kpx['RWing'])
        select_kp_ids['torso'].append(kpname2kpx['LLeg'])
        select_kp_ids['torso'].append(kpname2kpx['RLeg'])

        select_kp_ids['tail'] = []
        select_kp_ids['tail'].append(kpname2kpx['Tail'])
    return select_kp_ids
