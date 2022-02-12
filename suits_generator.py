import sys
def generator(folder_name ,num_examples):

    change_position_of_image = 0                #set to 0 to stop this change
    change_size_of_image = 0                    #set to 0 to stop this change
    change_strokesthickness_of_image =0        #set to 0 to stop this change
    change_orientation_of_image = 0             #set to 0 to stop this change
    import os
    newpath = folder_name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    from PIL import Image, ImageDraw, ImageFilter
    import random
    if change_strokesthickness_of_image == 0:
        im2 = Image.open('hearts.jpg')

        im1 = Image.open('blankpage.jpg')

        im3 = Image.open('clubs.jpg')

        im4 = Image.open('diamonds.jpg')

        im5 = Image.open('spades.jpg')
    else:
        im2 = Image.open('heartsss.jpg')

        im3 = Image.open('clubss.jpg')

        im5 = Image.open('spadess.jpg')

        im4 = Image.open('diamondsss.jpg')

        im1 = Image.open('blankpage.jpg')

    #im5 = im5.convert('L')
    val = num_examples
    for i in range(1,val+1):
        i = str(i)
        cardnumber = random.randint(1,4)
        if cardnumber == 1:
            x = random.randint(0,250)
            y = random.randint(0,250)
            size = random.randint(450,700)
            if change_size_of_image == 1:
                im1 = im1.resize((size, size))
            else:
                im1 = im1.resize((450, 450))
            back_im = im1.copy()
            if change_position_of_image == 1:
                back_im.paste(im2, (x, y))
            else:
                back_im.paste(im2, (0, 0))
            #back_im.save('C:/Users/abhis/Desktop/genertor/r565ocket_pillow_paste_pos.jpg', quality=95)
            resized_img = back_im.resize((25, 25))
            resized_img = resized_img.convert('L')
            if change_orientation_of_image == 1:
                resized_img = resized_img.rotate(90)
            resized_img.save(folder_name+'/'+i+'_H.jpg')
        if cardnumber == 2:
            x = random.randint(0,250)
            y = random.randint(0,250)
            size = random.randint(700,900)
            if change_size_of_image == 1:
                im1 = im1.resize((size, size))
            else:
                im1 = im1.resize((700, 700))
            back_im = im1.copy()
            if change_position_of_image == 1:
                back_im.paste(im5, (x, y))
            else:
                back_im.paste(im5, (0, 0))
           # back_im.save('C:/Users/abhis/Desktop/genertor/r565ocket_pillow_paste_pos.jpg', quality=95)
            resized_img = back_im.resize((25, 25))
            resized_img = resized_img.convert('L')
            if change_orientation_of_image == 1:
                resized_img = resized_img.rotate(90)
            resized_img.save(folder_name+'/'+i+'_S.jpg')
        if cardnumber == 3:
            x = random.randint(0,250)
            y = random.randint(0,250)
            size = random.randint(700,900)
            if change_size_of_image == 1:
                im1 = im1.resize((size, size))
            else:
                im1 = im1.resize((700, 700))
            back_im = im1.copy()
            if change_position_of_image == 1:
                back_im.paste(im3, (x, y))
            else:
                back_im.paste(im3, (0, 0))
           # back_im.save('C:/Users/abhis/Desktop/genertor/r565ocket_pillow_paste_pos.jpg', quality=95)
            resized_img = back_im.resize((25, 25))
            resized_img = resized_img.convert('L')
            if change_orientation_of_image == 1:
                resized_img = resized_img.rotate(90)
            resized_img.save(folder_name+'/'+i+'_C.jpg')
        if cardnumber == 4:
            x = random.randint(0,250)
            y = random.randint(0,250)
            size = random.randint(600,900)
            if change_size_of_image == 1:
                im1 = im1.resize((size, size))
            else:
                im1 = im1.resize((600, 600))
            back_im = im1.copy()
            if change_position_of_image == 1:
                back_im.paste(im4, (x, y))
            else:
                back_im.paste(im4, (0, 0))
           # back_im.save('C:/Users/abhis/Desktop/genertor/r565ocket_pillow_paste_pos.jpg', quality=95)
            resized_img = back_im.resize((25, 25))
            resized_img = resized_img.convert('L')
            if change_orientation_of_image == 1:
                resized_img = resized_img.rotate(90)
            resized_img.save(folder_name+'/'+i+'_D.jpg')


i = 0
for arg in sys.argv:

	globals()["arg" + str(i)] = arg
	i = i+1

folder_name  = arg1
num_examples  = int(arg2)

generator(folder_name,num_examples)