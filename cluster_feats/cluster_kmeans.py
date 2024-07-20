import numpy as np
import pickle
from libKMCUDA import kmeans_cuda

def kmeans():
    np.random.seed(42)
    for cluster_num in range(10, 110, 10):
        save_centroids = []
        with open('features/res50_SER.pkl', 'rb') as f:
            emotion_features = pickle.load(f)
            for i in range(len(emotion_features)):
                emotion_feat = emotion_features[i]
                centroids, assignments = kmeans_cuda(emotion_feat, 
                                                    clusters=cluster_num, 
                                                    verbosity=1, 
                                                    seed=42)
                if len(save_centroids)==0:
                    save_centroids = centroids
                else:
                    save_centroids = np.concatenate([save_centroids, centroids], axis=0)
                
            save_centroids = np.nan_to_num(save_centroids)
            np.save(f'cluster_{cluster_num}.npy', save_centroids)

if __name__=='__main__':
    kmeans()