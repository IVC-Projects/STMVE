#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('listfile', help='one line should contain paths "img0.ext img1.ext out.flo"')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()
# args.listfile = r'/media/jiaby/Elements/MF/linxi'


if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
if(not os.path.exists(args.listfile)): raise BaseException('listfile does not exist: '+args.listfile)

def readTupleList(filename):
    list = []
    for line in open(filename).readlines():
        if line.strip() != '':
            list.append(line.split())

    return list

ops = readTupleList(args.listfile)
finalpath=r'/media/jiaby/Elements/TJC/FLOW/BasketballDrill'
finalpath_png=r'/media/jiaby/Elements/TJC/FLOW/BasketballDrill'
width = -1
height = -1
pages=0
page_size=400
count=0
def getYdata(path, size):
    Yt={}
    w= size[0]
    h=size[1]
    width1=w%page_size
    height1=h%page_size
    if w%page_size!=0:
        page1=w/page_size+1
    else:
        page1=w/page_size
    if h%page_size!=0:
        page2=h/page_size+1
    else:
        page2=h/page_size
    pages=page1*page2
    
    with open(path, 'rb') as fp:
        org_data=np.zeros([ h, w], dtype="uint8",order='c')
        for n in range(h):
            for m in range(w):
                org_data[n,m]=ord(fp.read(1))
        k=0
        if page1==1 and page2==1:
            Yt[k]=org_data[:h,:w]
        elif page1==1:
            for m in range(page2):
                if m==0:
                    Yt[k]=org_data[:page_size+100,:w]
                elif m==page2-1:
                    Yt[k]==org_data[m*page_size-100:h,:w]
                else:
                    Yt[k]=org_data[m*page_size-100:(m+1)*page_size+100,:w]
                k=k+1
        elif page2==1:
            for m in range(page1):
                if m==0:
                    Yt[k]=org_data[:h,:page_size+100]
                elif m==page2-1:
                    Yt[k]=org_data[:h,m*page_size-100:w]
                else:
                    Yt[k]=org_data[:h,m*page_size-100:(m+1)*page_size+100]
                k=k+1
        else:
            for n in range(page2):
                for m in range(page1):
                    if n==0:
                        if m==0:
                            Yt[k]=org_data[0:page_size+100,:page_size+100]
                        elif m==page1-1:
                            Yt[k]=org_data[:page_size+100,m*page_size-100:w]
                        else:
                            Yt[k]=org_data[:page_size+100,m*page_size-100:(m+1)*page_size+100]
                    elif n==page2-1:
                        if m==0:
                            Yt[k]=org_data[page_size*n-100:h,:page_size+100]
                        elif m==page1-1:
                            Yt[k]=org_data[page_size*n-100:h,m*page_size-100:w]
                        else:
                            Yt[k]=org_data[page_size*n-100:h,m*page_size-100:(m+1)*page_size+100]
                    elif m==0:
                        Yt[k]=org_data[page_size*n-100:(n+1)*page_size+100,:(m+1)*page_size+100]
                    elif m==page1-1:
                        Yt[k]=org_data[page_size*n-100:(n+1)*page_size+100,m*page_size-100:w]
                    else:
                        Yt[k]=org_data[page_size*n-100:(n+1)*page_size+100,m*page_size-100:(m+1)*page_size+100]
                    k=k+1
    # for m in range(pages):
    return Yt
for ent in ops:
    print('Processing tuple:', ent)

    num_blobs = 2
    #img0 = misc.imread(ent[0])
    filename=os.path.basename(ent[0]).split('.')[0]
    filename1=os.path.basename(ent[1]).split('.')[0]
    print(filename)
    # hxw=filename.split('_')[1]
    hxw=filename.split('_')[-2]
    w,h=hxw.split('x')
    h=int(h)
    w=int(w)
    print(h,w)
    width1=w%page_size
    height1=h%page_size
    if w%page_size!=0:
        page1=w/page_size+1
    else:
        page1=w/page_size
    if h%page_size!=0:
        page2=h/page_size+1
    else:
        page2=h/page_size
    pages=page1*page2
    img0_0=getYdata(ent[0],[w,h])
    img0_1=getYdata(ent[1],[w,h])
    count+=1
    data={}
    print(pages)
    for m in range(pages):
        input_data = []
        img0=img0_0[m]
        if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        img1=img0_1[m]
        if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

        if width != input_data[0].shape[3] or height != input_data[0].shape[2]:
            width = input_data[0].shape[3]
            height = input_data[0].shape[2]

            vars = {}
            vars['TARGET_WIDTH'] = width
            vars['TARGET_HEIGHT'] = height

            divisor = 64.
            vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
            vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

            vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
            vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

            proto = open(args.deployproto).readlines()
            for line in proto:
                for key, value in vars.items():
                    tag = "$%s$" % key
                    line = line.replace(tag, str(value))

                tmp.write(line)

            tmp.flush()
        print(m,width,height)

        if not args.verbose:
            caffe.set_logging_disabled()
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()
        net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

    #
    # There is some non-deterministic nan-bug in caffe
    #
        print('Network forward pass using %s.' % args.caffemodel)
        i = 1
        while i<=5:
            i+=1

            net.forward(**input_dict)

            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
                print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')

        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)

        def readFlow(name,name1):
            if name.endswith('.pfm') or name.endswith('.PFM'):
                return readPFM(name)[0][:,:,0:2]

            f = open(name, 'rb')
            header = f.read(4)
            if header.decode("utf-8") != 'PIEH':
                raise Exception('Flow file header does not contain PIEH')

            width = np.fromfile(f, np.int32, 1).squeeze()
            height = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, width * height * 2)
            f.flush()
            f.close()
    # f1 = open(name1, 'rb')
    # # image_data= np.fromfile(f1, "uint8",15)
    # image_data= np.fromfile(f1, "uint8")
            i=0
    # image_data=flow
            image_data={}
            for n in range(height):
                for m in range(width):
                    image_data[i]=name1[n][m]
                    i+=1
    #print(image_data)
    # image_data= np.fromfile(f1, "uint8", width * height)
            warped_data = {}

            y = 0
            while y < height:
                x = 0
                while x < width:
                    fx = flow[y*width*2+x*2]
                    fy = flow[y*width*2+x*2+1]
                    x2= float(x)+fx
                    y2= float(y)+fy
                    x3= int(x2)
                    y3= int(y2)
                    #print(flow.shape,image_data.shape,x2,y2,x3,y3,x,y)
                    if x2<=0 and y2>0 and y2<height-1:
                        warped_data[y*width+x]=(y3+1-y2)*image_data[y3*width]+(y2-y3)*image_data[y3*width+width]
                    elif x2>0 and y2<=0 and x2<width-1:
                        warped_data[y*width+x]=(x3+1-x2)*image_data[x3]+(x2-x3)*image_data[x3+1]
                    elif x2>=width-1 and y2<height-1 and y2>0:
                        warped_data[y*width+x]=(y3+1-y2)*image_data[y3*width+width-1]+(y2-y3)*image_data[y3*width+width+width-1]
                    elif x2>0 and y2>=height-1 and x2<width-1:
                        warped_data[y*width+x]=(x3+1-x2)*image_data[width*height-width+x3]+(x2-x3)*image_data[width*height-width+x3+1]   
                    elif x2<=0 and y2<=0:
                        warped_data[y*width+x]=image_data[0]
                    elif x2>=width-1 and y2<=0:
                        warped_data[y*width+x]=image_data[width-1]
                    elif x2>=width-1 and y2>=height-1:
                        warped_data[y*width+x]=image_data[width*height-1]
                    elif x2<=0 and y2>=height-1:
                        warped_data[y*width+x]=image_data[width*height-width]
                    elif x2>0 and x2<width-1 and y2>0 and y2<height-1:
                        ix2_L = int(x2)
                        iy2_T = int(y2) # integer number.
                        ix2_R = min(ix2_L+1,width-1)
                        iy2_B = min(iy2_T+1,height-1)
                        alpha=x2-ix2_L
                        beta=y2-iy2_T
                        # print(ix2_L,iy2_T,ix2_R,iy2_B)
                        # print(iy2_T*width + ix2_L)
                        TL = image_data[iy2_T*width + ix2_L] # the pix of Top Left position.
                        TR = image_data[iy2_T*width + ix2_R] # the ... of Top right ...
                        BL = image_data[iy2_B*width + ix2_L] # the ... of Bootom Left
                        BR = image_data[iy2_B*width + ix2_R] # the ... of Bootom Right.
                        # print(y*width + x)
                        warped_data[y*width + x] =(1-alpha)*(1-beta)*TL +alpha*(1-beta)*TR +(1-alpha)*beta*BL +alpha*beta*BR
                    x+=1
                y+=1
            i=0
            while i<height*width-1:
                warped_data[i]=int(warped_data[i])
                i+=1
            i=0
            img = np.zeros([height, width], dtype="uint8", order='C')
            # convert to multi-dimensions.
            for n in range(height):
                for m in range(width):
                    img[n][m]=warped_data[i]
                    i+=1
            return img

        def writeFlow(name, flow):
            f = open(name, 'wb')
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)

        writeFlow(ent[2], blob)
        data[m]=readFlow(ent[2],img1)
        print(data[m].shape)
    final_data = np.zeros([h, w], dtype="uint8", order='C')
    if page1==1 and page2==1:
        final_data=data[0]
    elif page1==1:
        for m in range(page2):
            if m==0:
                data1=data[m]
                data1=data1[:page_size,:w]
                final_data[:page_size,:]=data1
            elif m==page2-1:
                data1=data[m]
                data1=data1[100:h-page_size*m+100,:w]
                final_data[page_size*m:h,:]=data1
            else:
                data1=data[m]
                data1=data1[100:page_size+100,:w]
                final_data[page_size*m:page_size*(m+1),:]=data1
    elif page2==1:
        for m in range(page1):
            if m==0:
                data1=data[m]
                data1=data1[:h,:page_size]
                final_data[:,:page_size]=data1
            elif m==page2-1:
                data1=data[m]
                data1=data1[:h,100:w-page_size*m+100]
                final_data[:,page_size*m:w]=data1
            else:
                data1=data[m]
                data1=data1[:h,100:100+page_size]
                final_data[:,page_size*m:page_size*(m+1)]=data1
    else:
        for n in range(page2):
            for m in range(page1):
                if n==0:
                    if m==0:
                        data1=data[page1*n+m]
                        # data1=data1[a]
                        data1=data1[:page_size,:page_size]
                        final_data[:page_size*(n+1),:page_size*(m+1)]=data1
                    elif m==page1-1:
                        data1=data[page1*n+m]
                        # data1=data1[a]
                        data1=data1[:page_size,100:w-page_size*m+100]
                        final_data[:page_size*(n+1),page_size*m:page_size*(m+1)]=data1
                    else:
                        data1=data[page1*n+m]
                        # data1=data1[a]
                        data1=data1[:page_size,100:page_size+100]
                        final_data[:page_size*(n+1),page_size*m:page_size*(m+1)]=data1
                elif n==page2-1:
                    if m==0:
                        data1=data[page1*n+m]
                        # data1=data1[a]
                        data1=data1[100:h-n*page_size+100,:page_size]
                        final_data[page_size*n:h,:page_size*(m+1)]=data1
                    elif m==page1-1:
                        data1=data[page1*n+m]
                        # data1=data1[a]
                        data1=data1[100:h-n*page_size+100,100:w-m*page_size+100]
                        final_data[page_size*n:h,page_size*m:w]=data1
                    else:
                        data1=data[page1*n+m]
                        # data1=data1[a]
                        data1=data1[100:h-page_size*n+100,100:page_size+100]
                        final_data[page_size*n:h,page_size*m:page_size*(m+1)]=data1
                elif m==0:
                    data1=data[page1*n+m]
                    # data1=data1[a]
                    data1=data1[100:page_size+100,:page_size]
                    final_data[page_size*n:page_size*(n+1),:page_size]=data1
                elif m==page1-1:
                    data1=data[page1*n+m]
                    # data1=data1[a]
                    data1=data1[100:page_size+100,100:w-m*page_size+100]
                    final_data[page_size*n:page_size*(n+1),page_size*m:w]=data1
                else:
                    data1=data[page1*n+m]
                    # data1=data1[a]
                    data1=data1[100:page_size+100,100:page_size+100]
                    final_data[page_size*n:page_size*(n+1),page_size*m:page_size*(m+1)]=data1
    print(final_data.shape,count)
    final_data.tofile(os.path.join(finalpath,filename1)+".yuv")
    im = Image.fromarray(final_data)
    im.save(os.path.join(finalpath_png,filename1)+".png")

