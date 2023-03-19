#This project is to analyze how many valid bounding boxes are there in the annotate.txt file.
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

if __name__ == '__main__':
    anno_file = 'annotation.txt'
    with open(anno_file, 'r') as fp:
        lines = fp.readlines()

    # Enter your code here (25%)
    # You might use the following to access each line
    width_list = []
    widthheight_list = []
    invalid_count=0
    
    valid_count_1=0
    valid_count_2=0
    valid_count_3=0
    valid_count_4=0
    valid_count_5=0
    valid_count_6=0
    valid_count_7=0
    valid_count_8=0
    valid_count_9=0
    valid_count_10=0
    valid_count_11=0
    valid_count_12=0
    
    # for line in lines:
    for line in lines:
        annotation = line.strip().split(' ')
        integer = np.array(annotation[1:], dtype="int32")
        list = np.reshape(integer, (-1,4))
        for i in range(list.shape[0]):
            a = list[i, 0]
            b = list[i, 1]
            c = list[i, 2]
            d = list[i, 3]
            if a>=0 and b>=0 and c>a and d>b:
                width = c - a + 1
                height = d - b + 1
                width_list.append(width)
                widthheight_list.append(width/height)   
            else:
                invalid_count=invalid_count+1
                
    for width in width_list:
        if width<10:
            valid_count_1=valid_count_1+1
        elif 10<=width<20:
            valid_count_2=valid_count_2+1
        elif 20<=width<30:
            valid_count_3=valid_count_3+1 
        elif 30<=width<40:
            valid_count_4=valid_count_4+1
        elif 40<=width<50:
            valid_count_5=valid_count_5+1
        elif width>=50:
            valid_count_6=valid_count_6+1
    
    
    for widthheight in widthheight_list:
        if widthheight<0.6:
            valid_count_7=valid_count_7+1
        elif 0.6<=widthheight<0.7:
            valid_count_8=valid_count_8+1
        elif 0.7<=widthheight<0.8:
            valid_count_9=valid_count_9+1 
        elif 0.8<=widthheight<0.9:
            valid_count_10=valid_count_10+1
        elif 0.9<=widthheight<1.0:
            valid_count_11=valid_count_11+1
        elif widthheight>=1.0:
            valid_count_12=valid_count_12+1
            
    print("      width < 10:", valid_count_1,"\n", 
          "10 <= width < 20:", valid_count_2,"\n",
          "20 <= width < 30:", valid_count_3,"\n",
          "30 <= width < 40:", valid_count_4,"\n",
          "40 <= width < 50:", valid_count_5,"\n",
          "50 <= width     :", valid_count_6,"\n",)
    
    print("       width/height < 0.6:", valid_count_7,"\n", 
          "0.6 <= width/height < 0.7:", valid_count_8,"\n",
          "0.7 <= width/height < 0.8:", valid_count_9,"\n",
          "0.8 <= width/height < 0.9:", valid_count_10,"\n",
          "0.9 <= width/height < 1.0:", valid_count_11,"\n",
          "1.0 <= width/height      :", valid_count_12,"\n",)
    
    print("invalid bounding boxes:",invalid_count)
    
    plt.hist(width_list, range(0,101,10), rwidth=0.85, color="slateblue")
    plt.ylabel("Counts")
    plt.xlabel("Width")     
    plt.grid(axis='y',linestyle='-')
      

      
