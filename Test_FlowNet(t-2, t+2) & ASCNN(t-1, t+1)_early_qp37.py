import time
import itertools
from FlowNet(t-2, t+2) & ASCNN(t-1, t+1)_early_qp37_network import network  # double model FlowNet04+ASCNN13
from FlowNet(t-2, t+2) & ASCNN(t-1, t+1)_early_qp37_network import network  # single model
from UTIL_FlowNet(t-2, t+2) & ASCNN(t-1, t+1)_early_qp37 import *



tf.logging.set_verbosity(tf.logging.WARN)

# There setting should be modified by you, and you should change the 112-lines about "epoch number" according to the number in ckpt-file name.
EXP_DATA1 = "MF_qp37_ra_0512_ResNet7_AS13_Flow04_earlyFusion"  # double
EXP_DATA2 = "MF_qp37_ra_0512_ResNet7_AS13_Flow04_earlyFusion"
MODEL_DOUBLE_PATH1 = "./checkpoints/%s/"%(EXP_DATA1)
MODEL_DOUBLE_PATH2 = "./checkpoints/%s/"%(EXP_DATA2)

ASCNN_Frames_PATH = r"E:\MF\HEVC_TestSequenct\ASCNN_out\qp37_13\ra\D\BQSquare_qp22_ai_416x240_yuv"
HIGHDATA_Parent_PATH = r"E:\MF\HEVC_TestSequenct\test\qp37\ra\D\BQSquare_qp37_ra_416x240"
QP_LOWDATA_PATH = r'E:\MF\HEVC_TestSequenct\rec\qp37\ra\D\BQSquare_qp37_ra_416x240'
GT_PATH = r"E:\MF\HEVC_TestSequenct\org\D\BQSquare_416x240"
DL_path = r'E:\MF\HEVC_TestSequenct\DL_rec\RseNet4_Slim_ASCNN13_Flow04_slowFusion'
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA1)
NOFILTER = {'q22':42.2758, 'q27':38.9788, 'qp32':35.8667, 'q37':32.8257,'qp37':32.8257}


def prepare_test_data(fileOrDir):
    doubleData_ycbcr = []
    doubleGT_y = []
    singleData_ycbcr = []
    singleGT_y = []

    fileName_list = []
    #The input is a single file.
    if len(fileOrDir) == 4:

        fileName_list = load_file_list(fileOrDir[2])

        double_list, single_list = get_test_list(HIGHDATA_Parent_PATH, ASCNN_Frames_PATH,
                                                 load_file_list(fileOrDir[2]), load_file_list(fileOrDir[3]))

        for pair in double_list:
            high1Data_List = []
            lowData_List = []
            high2Data_List = []
            ascnnData_List = []

            high1Data_imgY = c_getYdata(pair[0][0])
            high2Data_imgY = c_getYdata(pair[0][1])
            lowData_imgY = c_getYdata(pair[1])
            CbCr = c_getCbCr(pair[1])
            ascnn_imgY = c_getYdata(pair[2])
            gt_imgY = c_getYdata(pair[3])

            #normalize
            high1Data_imgY = normalize(high1Data_imgY)
            lowData_imgY = normalize(lowData_imgY)
            high2Data_imgY = normalize(high2Data_imgY)
            ascnn_imgY = normalize(ascnn_imgY)

            high1Data_imgY = np.resize(high1Data_imgY, (1, high1Data_imgY.shape[0], high1Data_imgY.shape[1],1))
            lowData_imgY = np.resize(lowData_imgY, (1, lowData_imgY.shape[0], lowData_imgY.shape[1], 1))
            high2Data_imgY = np.resize(high2Data_imgY, (1, high2Data_imgY.shape[0], high2Data_imgY.shape[1], 1))
            ascnn_imgY = np.resize(ascnn_imgY, (1, ascnn_imgY.shape[0], ascnn_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1],1))

            ## act as a placeholder

            high1Data_List.append([high1Data_imgY, 0])
            lowData_List.append([lowData_imgY, CbCr])
            high2Data_List.append([high2Data_imgY, 0])
            ascnnData_List.append([ascnn_imgY, 0])
            doubleData_ycbcr.append([high1Data_List, lowData_List, high2Data_List, ascnnData_List])
            doubleGT_y.append(gt_imgY)

    else:
        print("Invalid Inputs...!tjc!")
        exit(0)

    return doubleData_ycbcr, doubleGT_y, singleData_ycbcr, singleGT_y, fileName_list

def abc(modelPath1, modelPath2, fileOrDir):
    max = [0, 0]

    tem1 = [f for f in os.listdir(modelPath1) if 'data' in f]
    ckptFiles1 = sorted([r.split('.data')[0] for r in tem1])

    tem2 = [f for f in os.listdir(modelPath2) if 'data' in f]
    ckptFiles2 = sorted([r.split('.data')[0] for r in tem2])

    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)


    doubleData_ycbcr, doubleGT_y, singleData_ycbcr, singleGT_y, fileName_list = prepare_test_data(fileOrDir)
    total_time, total_psnr = 0, 0
    total_imgs = len(fileName_list)
    count = 0
    for i in range(total_imgs):
        if i % 4 != 0:
            count += 1
            j = i - (i//4) - 1
            imgHigh1DataY = doubleData_ycbcr[j][0][0][0]
            imgLowDataY = doubleData_ycbcr[j][1][0][0]
            imgLowCbCr = doubleData_ycbcr[j][1][0][1]
            imgHigh2DataY = doubleData_ycbcr[j][2][0][0]
            imgAscnnDataY = doubleData_ycbcr[j][3][0][0]
            #imgCbCr = original_ycbcr[i][1]
            gtY = doubleGT_y[j] if doubleGT_y else 0
            start_t = time.time()
            for ckpt1 in ckptFiles1:
                epoch = int(ckpt1.split('_')[-1].split('.')[0])
                if epoch != 17:
                    continue
                # very important!!!!!!!
                tf.reset_default_graph()
                # Double section
                high1Data_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                lowData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                high2Data_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                ascnnData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                shared_model1 = tf.make_template('shared_model', network)
                output_tensor1 = shared_model1(high1Data_tensor, lowData_tensor, high2Data_tensor, ascnnData_tensor)
                # output_tensor = shared_model(input_tensor)
                output_tensor1 = tf.clip_by_value(output_tensor1, 0., 1.)
                output_tensor1 = output_tensor1 * 255
                with tf.Session() as sess:
                    saver = tf.train.Saver(tf.global_variables())
                    sess.run(tf.global_variables_initializer())

                    saver.restore(sess, os.path.join(modelPath1, ckpt1))
                    out = sess.run(output_tensor1, feed_dict={high1Data_tensor: imgHigh1DataY, lowData_tensor: imgLowDataY,
                                                              high2Data_tensor: imgHigh2DataY, ascnnData_tensor: imgAscnnDataY})
                    hevc = psnr(imgLowDataY * 255.0, gtY)
                    # print(hevc)
                    out = np.around(out)
                    out = out.astype('int')
                    out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])

                    Y = np.reshape(out, [out.shape[1], out.shape[2]])
                    Y = np.array(list(itertools.chain.from_iterable(Y)))
                    U = imgLowCbCr[0]
                    V = imgLowCbCr[1]
                    creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                    if not os.path.exists(creatPath):
                        os.mkdir(creatPath)

                    if doubleGT_y:
                        p = psnr(out, gtY)

                        path = os.path.join(DL_path,
                                            fileName_list[i].split('\\')[-2],
                                            fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p-hevc)+ '.yuv'

                        YUV = np.concatenate((Y, U, V))
                        YUV = YUV.astype('uint8')
                        YUV.tofile(path)

                        total_psnr += p
                        print("qp37\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))

            duration_t = time.time() - start_t
            total_time += duration_t
        else:
            continue
            count += 1
            j = i // 4
            ## ???
            lowDataY = singleData_ycbcr[j][0][0]
            imgLowCbCr = singleData_ycbcr[j][0][1]
            # imgCbCr = original_ycbcr[i][1]
            gtY = singleGT_y[j] if singleGT_y else 0

            hevc = psnr(lowDataY * 255.0, gtY)
            # print(hevc)

            start_t = time.time()
            for ckpt2 in ckptFiles2:
                epoch = int(ckpt2.split('_')[-1].split('.')[0])
                if epoch != 169:
                    continue

                tf.reset_default_graph()
                # Single section
                lowSingleData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                shared_model2 = tf.make_template('shared_model', model_single)
                output_tensor2 = shared_model2(lowSingleData_tensor)
                # output_tensor = shared_model(input_tensor)
                output_tensor2 = tf.clip_by_value(output_tensor2, 0., 1.)
                output_tensor2 = output_tensor2 * 255
                with tf.Session() as sess:
                    saver = tf.train.Saver(tf.global_variables())
                    sess.run(tf.global_variables_initializer())

                    saver.restore(sess, os.path.join(modelPath2, ckpt2))
                    out = sess.run(output_tensor2, feed_dict={lowSingleData_tensor: lowDataY})
                    out = np.around(out)
                    out = out.astype('int')
                    out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])
                    Y = np.reshape(out, [out.shape[1], out.shape[2]])
                    Y = np.array(list(itertools.chain.from_iterable(Y)))
                    U = imgLowCbCr[0]
                    V = imgLowCbCr[1]
                    creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                    if not os.path.exists(creatPath):
                        os.mkdir(creatPath)


                    if singleGT_y:
                        p = psnr(out, gtY)

                        path = os.path.join(DL_path, fileName_list[i].split('\\')[-2],
                                            fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p - hevc) + '.yuv'

                        YUV = np.concatenate((Y, U, V))
                        YUV = YUV.astype('uint8')
                        YUV.tofile(path)


                        total_psnr += p
                        print("qp37\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))

            duration_t = time.time() - start_t
            total_time += duration_t


    print("AVG_DURATION:%.2f\tAVG_PSNR:%.4f"%(total_time/total_imgs, total_psnr / count))
    print('count:', count)
    # avg_psnr = total_psnr/total_imgs
    avg_psnr = total_psnr / count
    avg_duration = (total_time/total_imgs)
    if avg_psnr > max[0]:
        max[0] = avg_psnr
        max[1] = epoch

if __name__ == '__main__':
    abc(MODEL_DOUBLE_PATH1, MODEL_DOUBLE_PATH2, [HIGHDATA_Parent_PATH, ASCNN_Frames_PATH, QP_LOWDATA_PATH, GT_PATH])
