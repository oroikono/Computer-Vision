import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    #desc1 is of shape 1999x81 while desc2 is of shape 2300x81 so we create a bix matrix 
    # that contains of all the differences  for the pairs of descriptors
    differences = []
    differences = desc1[:,np.newaxis,:]-desc2[np.newaxis]
    
    #find the square for each potential difference
    differences_2 = np.square(differences)
    return(np.sum(differences_2 ,axis=-1))

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    
    distances = ssd(desc1, desc2)
    #print(len(distances))
    
    q1, q2 = desc1.shape[0], desc2.shape[0]

    
    matches = None

    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        
        #extract the feature with that has the smallest distance between the two images
        min_feature_dist = np.argmin(distances, 1)
        #array that contains the index of the descriptor and its chosen feature
        matches = np . array ([[ i , dist ] for i , dist in zip(range(q1), min_feature_dist)])

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here

        #extract the feature with that has the smallest distance between the two images I1 to I2
        min_feature_dist12 = np.argmin(distances, 1)
        #array that contains the index of the descriptor and its chosen feature
        matchesI1I2 = np . array ([[ i , dist ] for i , dist in zip(range(q1), min_feature_dist12)])
        
        #extract the feature with that has the smallest distance between the two images I2 to I1
        min_feature_dist21 = np.argmin(distances,0)
        #array that contains chosen feature of the descriptor and its index
        matchesI2I1 = np . array ([[ dist , i ] for i , dist in zip(range(q2), min_feature_dist21)])

        # chosen descriptor feature I2 to I1
        chDF = matchesI2I1[:, 0]
        
        # boolean array that contains all the cases that matches from I1 to I2
        # are matches from I2 to I1 too
        mutual = matchesI1I2[chDF, 1] == matchesI2I1[:, 1]

        # as in this case each matching might give different number of matches
        # we ensure that the array of mutual matches is the same size with
        # the array of matches that we will use to extract the final info
        if (len(matchesI1I2)< len(mutual)):
            mutual = mutual[:len(matchesI1I2)]
            matches = matchesI1I2[mutual]
        else:
            matches = matchesI1I2[mutual]
        print(matchesI2I1)

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        
         # get the first and second neighbor
        partition = np.partition (distances,(0,1),1 )

        #apply ratio threshold
        partition_ratio = partition[:,0]/partition[:,1] < ratio_thresh

        #find min distances for all the feature pairs
        match = np.arange(q1)
        fea_min_dist = np.argmin(distances, 1)

        #initialization of array and assign zeros for ratios that
        #did not make through the threshold
        matches = np.zeros((q1,2))
        
        #use found indexes to the descriptors that we are going to keep
        col = match[partition_ratio]
        row = fea_min_dist[partition_ratio]
        matches = np.array([col,row]).T
 
    else:
        raise NotImplementedError
    return matches


