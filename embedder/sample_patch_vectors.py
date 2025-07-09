import h5py
import os
import pandas as pd
import numpy as np
import pickle
import argparse



def get_data(idx,feat_dir,metadata_df):
    slide_id = str(metadata_df['slide_id'][idx])
    label = str(metadata_df['label'][idx])

    full_path = os.path.join(feat_dir,'h5_files','{}.h5'.format(slide_id))
    with h5py.File(full_path,'r') as hdf5_file:
        features = hdf5_file['features'][:]

    return slide_id, features, label


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()
     
def do_one(feat_dir,metadata_csv,output_path,feat_size):
    n_samples=10
    metadata_df = pd.read_csv(metadata_csv)
    mode='w'
    rng = np.random.default_rng()
    asset_dict = {"slide_id":[], 'features': np.zeros((len(metadata_df)*n_samples,feat_size)), 'label':[]}
    print("Starting...")
    for i in range(len(metadata_df)):
        sid, feats, lab = get_data(i,feat_dir,metadata_df)
        # print(sid)
        s = rng.integers(low=0,high=len(feats),size=n_samples)
        feats=feats[s]
        #sid = [sid]*n_samples; lab=[lab]*n_samples
        asset_dict['slide_id'].append(sid)
        asset_dict['label'].append(lab)
        asset_dict['features'][i*n_samples:i*n_samples+n_samples,:]=feats
    save_pkl(output_path,asset_dict)


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--embedder', type=str)
args = parser.parse_args()

if __name__=="__main__":
    augments = ['pp','rc-mix-wsi','macenko-norm-gbm']#['rc-mix-wsi','macenko-norm-gbm']
    partitions=['ebrains_all']
    base_feat_dir = "/mnt/data/"+args.embedder+"_20x/"
    if(args.embedder=='uni'):
        feat_size=1024
    elif(args.embedder=='gigapath'):
        feat_size=1536
    elif(args.embedder=='virchow'):
        feat_size=2560
    else:
        raise NotImplementedError()

    for aug in augments:
         feat_dir=base_feat_dir+aug
         for p in partitions:
            print(aug,p)
            metadata_csv='/mnt/metadata/'+p+'.csv'
            output_path=os.path.join('output',f'sample_feature_vecs_{p}_{args.embedder}_{aug}.pkl')
            try:
                do_one(feat_dir,metadata_csv,output_path,feat_size)
            except Exception as e:
                print("Failed, skipping")
                print(e)
                continue

    print("Finished!")

