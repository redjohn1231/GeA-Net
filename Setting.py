# -*- coding: utf-8 -*-

class dataset:
    
    path_train = 'E:/DATA_new/Dtrain/'
    path_val   = 'E:/DATA_new/Dval/'
    path_test  = 'E:/DATA_new/Dtest/'
    path_fit   = 'E:/DATA_new/Dfit/'
    
    
    name = ['/(1)FaceSwap(FF++)/',
            '/(2)NeuralTextures(FF++)/',
            '/(3)FaceAPP(DFFD)/',
            '/(4)StyleGAN(DFFD)/',
            '/(5)StyleGAN/']
    
    name_1 = ['/(1)DeepFake(FF++)/',
              '/(2)FaceSwap(FF++)/',
              '/(3)Face2Face(FF++)/',
              '/(4)NeuralTextures(FF++)/',
              '/(5)FaceAPP(DFFD)/',
              '/(6)StarGAN(DFFD)/',
              '/(7)StyleGAN(DFFD)/',
              '/(8)PGGAN(DFFD)/',
              '/(9)DeepFake(Celeb-DF)/',
              '/(10)StyleGAN/',
              '/(11)StyleGAN2/',
              '/(12)DFDC/', 
              '/(13)DreamBooth(DiFF)/',
              '/(14)CoDiff(DiFF)/',
              '/(15)DCFace(DiFF)/']
    
    
    name_s = [name_1[1],
              name_1[3],
              name_1[4],
              name_1[6],
              name_1[9]]

    
    name_d = [name_1[0],
              name_1[2],
              name_1[5],
              name_1[7],
              name_1[8],
              name_1[10],
              name_1[11],
              name_1[12],
              name_1[13],
              name_1[14]]
    
    #D_train
    path_train_0 = []
    path_train_1 = []
    for i in range(len(name)):
        path_train_0.append(path_train + name[i] + '/FAKE/')
    for i in range(len(name)):
        path_train_1.append(path_train + name[i] + '/REAL/')
    
    #D_val
    path_val_0 = []
    path_val_1 = []
    for i in range(len(name)):
        path_val_0.append(path_val + name[i] + '/FAKE/')
    for i in range(len(name)):
        path_val_1.append(path_val + name[i] + '/REAL/')
        
    #D_fit
    path_fit_0 = []
    path_fit_0_1 = []
    for i in range(len(name)):
        path_fit_0.append(path_fit + name[i] + '/FAKE/')
    for i in range(len(name)):
        path_fit_0_1.append(path_fit  + name[i] + '/REAL/')
        
    #D_test_in
    path_test_in_0 = []
    path_test_in_1 = []
    for i in range(len(name_s)):
        path_test_in_0.append(path_test + name_s[i] + '/FAKE/')
    for i in range(len(name_s)):
        path_test_in_1.append(path_test + name_s[i] + '/REAL/')
        
    #D_test_cross
    path_test_cross_0 = []
    path_test_cross_1 = []
    for i in range(len(name_d)):
        path_test_cross_0.append(path_test + name_d[i] + '/FAKE/')
    for i in range(len(name_d)):
        path_test_cross_1.append(path_test + name_d[i] + '/REAL/')



        