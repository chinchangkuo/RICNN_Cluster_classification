import cv2
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import os

##-------------input parameters------------------
#folder_dir = "D:\\Bubble cluster\\trixiedata\\image_1_1.48_g15_08_6"
#folder_dir = "D:\\Bubble cluster\\trixiedata\\image_1_1.48_g15_08_6\\3L3S\\allstate"
folder_dir = 'CNN_demo'
## The folder to save image
n_l = 3
n_s = 3
## The numbers of large and small bubbles to form the cluster
c_a_max = 20000
c_a_min = 500
## The area range for clusters in pixels
b_a_max = 1000
b_a_s_l = 300
b_a_min = 50
## The area range and the threshold for large and small bubbls in pixels 
CNN_w = 250
## The size of the input image for CNN in pixels. It needs to be adjusted by
## the cluster size

## modified for demo:  114, 127, 221, 215, 216

##-----------------------------------------------

def cuser_crop(img):
    h, w = img.shape
    imgs = cv2.resize(img, (int(w/2), int(h/2)))
    r = cv2.selectROI('Press Enter after the selection', imgs)
    cv2.destroyAllWindows()
    c_rect = (int(r[0])*2, int(r[1])*2 ,int(r[2])*2, int(r[3])*2)
    return  c_rect
    ##for select ROI to prevent the unstable forming cluster

def local_binary( imcrop, local_size, sub_mean ):
    ##bw = local_binary( imcrop, 151, 8 ) 
    bw = cv2.adaptiveThreshold(imcrop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,local_size,sub_mean)
    return bw
    ##Adaptive Thresholding, (original image, I_max, algorithm,binary_inv method,   
    ##                        neighbourhood area size, subtracted from the mean)
    ## Adjust last two parameters for different data.


def bubble_close( bw , kernel_d=30):
    ##Use this function to fill bubbles to detect the cluster 
    ##bubble_close( bw , 30 )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_d,kernel_d))
    bw_filled = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    return bw_filled

def cluster_detect( imcrop, bw_filled, c_a_max=20000, c_a_min=500 ):
    output = cv2.connectedComponentsWithStats(bw_filled, 4, cv2.CV_32S)
    #cimg = cv2.cvtColor(imcrop,cv2.COLOR_GRAY2BGR)
    ##debug plot
    cluster_table = []
    for j in output[2]:
        if j[4] > c_a_min and j[4] < c_a_max:
            cluster_table.append(j)
            #cv2.rectangle(cimg,(j[0],j[1]),(j[0]+j[2],j[1]+j[3]),(0,255,0),5)
    #plt.imshow(cimg)        
    ##debug plot
    return cluster_table

def cluster_filter(imcrop, bw, cluster_table, cluster_count, b_a_max, b_a_min, b_a_s_l, CNN_w, current_img):
    cimg0 = cv2.cvtColor(imcrop,cv2.COLOR_GRAY2BGR)
    for i in cluster_table:
        h, w = bw.shape
        if int(i[1])-10 > 0 and int(i[0])-10 > 0 and int(i[1]+i[3])+10 < h and int(i[0]+i[2])+10 < w:
            c_y = int(i[0])-10
            c_x = int(i[1])-10
            c_bw = bw[int(i[1])-10:int(i[1]+i[3])+10,
                          int(i[0])-10:int(i[0]+i[2])+10]   
            clustercrop = imcrop[int(i[1])-10:int(i[1]+i[3])+10,
                          int(i[0])-10:int(i[0]+i[2])+10]   
            inv_c_bw = cv2.bitwise_not(c_bw)
            cc = cv2.connectedComponentsWithStats(inv_c_bw, 4, cv2.CV_32S)
            cimg1 = cv2.cvtColor(clustercrop,cv2.COLOR_GRAY2BGR)
            bubble_table_s = []
            ## x ,y, r
            bubble_table_l = []
            ## x ,y, r
            for j in range(cc[0]):
                if cc[2][j][4] > b_a_s_l and cc[2][j][4] < b_a_max:
                    bubble_table_l.append([cc[3][j][0],cc[3][j][1],(cc[2][j][4]/3.14)**(1/2)])
                    cv2.circle(cimg1,(int(cc[3][j][0]),int(cc[3][j][1])),1,(255,0,0),2) 
                    cv2.circle(cimg0,(int(cc[3][j][0])+c_y,int(cc[3][j][1])+c_x),1,(255,0,0),2) 
                elif cc[2][j][4] > b_a_min and cc[2][j][4] < b_a_s_l:
                    bubble_table_s.append([cc[3][j][0],cc[3][j][1],(cc[2][j][4]/3.14)**(1/2)])        
                    cv2.circle(cimg1,(int(cc[3][j][0]),int(cc[3][j][1])),1,(0,0,255),2) 
                    cv2.circle(cimg0,(int(cc[3][j][0])+c_y,int(cc[3][j][1])+c_x),1,(0,0,255),2) 
#            plt.imshow(cimg1)
#            wmngr = plt.get_current_fig_manager()
#            wgeom = wmngr.window.geometry()
#            wx,wy,wdx,wdy = wgeom.getRect()
#            wmngr.window.setGeometry(int(cc[3][j][0])+c_y,wy-200,wdx, wdy) 
#            wmngr.resize(wdx/2,wdy/2)
#            plt.figure()
            ##---------------------the key condition for the targeting cluster--
            if len(bubble_table_l) == n_l and len(bubble_table_s) == n_s:
                cluster_count += 1

                ## ------------------------------------------------------------
                state_index = []
                if n_l == 3 and n_s ==3:
                    cimg1, state_index = label_nls_3_3(bubble_table_l, bubble_table_s, cimg1)
                ##---------------------------special case for n_s = 3 and n_l = 3

                cv2.rectangle(cimg0,(i[0]-10,i[1]-10),(i[0]+i[2]+10,i[1]+i[3]+10),(0,255,0),5)
                if state_index != []:
                    plt.imsave(folder_dir + '\\' + save_folder_name + '\\'
                           + str(cluster_count) +'_'+ current_img + '_' + str(state_index) +'.png', cimg1)
                c_side = clustercrop.copy()
                c_side[10:-10, 10:-10] = 0
                bg_v = int(np.true_divide(c_side.sum(),(c_side!=0).sum()))                  
                ## calculate the background value for four sides of the cluster 
                CNN_cluster = np.empty( (CNN_w,CNN_w),dtype = 'uint8') 
                CNN_cluster[:] = bg_v
                CNN_cluster[int((CNN_w - i[3]- 20)/2):int((CNN_w - i[3]- 20)/2)+ i[3] + 20,
                            int((CNN_w - i[2]- 20)/2):int((CNN_w - i[2]- 20)/2)+ i[2] + 20] = clustercrop
                
                cimg2 = cv2.cvtColor(CNN_cluster,cv2.COLOR_GRAY2BGR)
                if state_index != []:
                    plt.imsave(folder_dir + '\\' + save_folder_name + '\\ CNNinput\\'
                           + str(cluster_count) +'_'+ current_img +'_' + str(state_index) +'.png', cimg2)
       
            ##------------------------------------------------------------------                 
#    plt.imshow(cimg0)        
#    wmngr = plt.get_current_fig_manager()
#    wgeom = wmngr.window.geometry()
#    wx,wy,wdx,wdy = wgeom.getRect()
#    wmngr.window.setGeometry(wx,wy+150,wdx, wdy)     
#    plt.pause(3)
#    plt.close("all")
    
    return cluster_count

def label_nls_3_3(bubble_table_l, bubble_table_s, cimg1): 
   configurations ={1:[2,6,10,32,36,40], 
                    2:[6,8,10,10,20,24],
                    3:[5,9,10,20,24,40],
                    4:[6,9,9,20,20,44],
                    5:[5,5,14,24,24,36],
                    6:[5,6,13,20,28,36],
                    7:[5,5,10,32,40,40],
                    8:[5,6,9,32,36,44],
                    9:[5,8,10,13,20,28],
                    10:[9,9,12,14,24,24],
                    11:[5,10,12,13,20,24],
                    12:[5,9,12,20,24,44],
                    13:[5,9,12,20,28,40],
                    14:[5,8,13,20,28,40],
                    15:[5,8,13,24,24,40],
                    16:[8,9,9,20,24,44],
                    17:[5,8,9,36,36,44],
                    18:[5,8,9,36,40,40],
                    19:[8,8,8,40,40,40],
                    20:[8,8,12,24,24,44],
                    21:[8,8,9,13,24,28],
                    22:[8,8,8,10,10,10],
                    23:[2,10,10,20,20,40],
                    24:[8,9,9,10,20,28],
                    25:[6,6,14,24,24,36]}
   bubble_table_all = bubble_table_l + bubble_table_s      
   dist = []
   index_code = []
   #mean_l_r = np.mean([ii[2] for ii in bubble_table_l])
   #mean_s_r = np.mean([ii[2] for ii in bubble_table_s])
   #margin = 5
   #error = 2
   for i in bubble_table_all:
       n_l_l = 0
       n_l_s = 0
       n_s_s = 0
       for j in bubble_table_all:           
           if i != j:
              dist = (((i[0]-j[0])**2 + (i[1]-j[1])**2)**(1/2))
              if dist > 23 and dist <  31:
                  if i in bubble_table_l and j in bubble_table_s:
                      n_l_s += 1
                      cv2.line(cimg1,(int(i[0]),int(i[1])),(int(j[0]),int(j[1])),(0,255,0),1)
                  if i in bubble_table_s and j in bubble_table_l:
                      n_l_s += 1
                      cv2.line(cimg1,(int(i[0]),int(i[1])),(int(j[0]),int(j[1])),(0,255,0),1)                      
              if dist >  19 and dist <  25.5:
                  if i in bubble_table_s and j in bubble_table_s:
                      n_s_s += 1
                      cv2.line(cimg1,(int(i[0]),int(i[1])),(int(j[0]),int(j[1])),(0,0,255),1)              
              if dist > 29 and dist < 35:
                  if i in bubble_table_l and j in bubble_table_l:
                      n_l_l += 1
                      cv2.line(cimg1,(int(i[0]),int(i[1])),(int(j[0]),int(j[1])),(255,0,0),1)
                      
       index_code.append(16*n_l_l + 4*n_l_s + n_s_s)  
       
   state_index = [ii for ii in configurations if configurations[ii] == sorted(index_code)]
       
   return cimg1, state_index
       
#-----------------------------------------------
all_img = [i for i in os.listdir(folder_dir) if i.endswith(".bmp")]
save_folder_name = 'CNN_l' + str(n_l) + '_s' + str(n_s)
pathlib.Path(folder_dir + '\\' + save_folder_name).mkdir(parents=True, exist_ok=True)
pathlib.Path(folder_dir + '\\' + save_folder_name + '\\ CNNinput').mkdir(parents=True, exist_ok=True)  
##find all .bmp image data in the folder
cluster_count = 0
for i in range(len(all_img)):
#for i in range(500):
    #current_img = str(all_img[i][-8:-4])
    current_img = all_img[i][all_img[i].find('state'):-4]
    #if i % 100 == 0:
        #print (i)
    img_path = folder_dir + '\\' + all_img[i]
    ##creat the path for each image data
    img = cv2.imread(img_path,0)
    if i == 0:
        #c_rect = cuser_crop(img)
        c_rect = (384, 126, 1238, 690)
        ## Pre-selected ROI for demo
    imcrop = img[int(c_rect[1]):int(c_rect[1]+c_rect[3]),
                 int(c_rect[0]):int(c_rect[0]+c_rect[2])]
    bw = local_binary( imcrop, 301, 6 )
    bw_filled = bubble_close( bw , 30 )
    cluster_table = cluster_detect( imcrop, bw_filled, c_a_max, c_a_min )
    cluster_count = cluster_filter( imcrop, bw, cluster_table, cluster_count,
                                   b_a_max, b_a_min, b_a_s_l, CNN_w, current_img)

