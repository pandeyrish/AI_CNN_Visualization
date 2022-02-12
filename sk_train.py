import sys
def svmviewer(epsilon,max_updates,class_letter,model_file_name,train_folder_name):
    import os
    import os
    val = len(os.listdir(train_folder_name))
    if val < 1:
        nodata = "NO DATA"
        return print(nodata)
    from PIL import Image, ImageDraw, ImageFilter
    import random

    # symbols = ["H","C","S","D"]

    import numpy as np

    hearts = 0
    x = 0
    xplus = []
    ll = -1
    shape = class_letter
    if shape == "H":
        shape1 = "C"
        shape2 = "S"
        shape3 = "D"
    if shape == "C":
        shape1 = "H"
        shape2 = "S"
        shape3 = "D"
    if shape == "S":
        shape1 = "C"
        shape2 = "H"
        shape3 = "D"
    if shape == "D":
        shape1 = "C"
        shape2 = "S"
        shape3 = "H"

    for v in range(1, val + 1):

        try:
            img = Image.open(train_folder_name+'/' + str(v) + '_' + shape + '.jpg')
            im_matrix = np.array(img)
            im_matrix = im_matrix.flatten()
            im_matrix = im_matrix.astype('float64')
            globals()["im_matrix" + str(v)] = im_matrix / 255
            xplus.append(globals()["im_matrix" + str(v)])
            x = x + globals()["im_matrix" + str(v)]
            hearts = hearts + 1
        except:
            pass

    mplus = x / hearts

    nonhearts = 0
    xminus = []
    x = 0
    for v in range(1, val + 1):
        try:
            try:
                img = Image.open(train_folder_name+'/' + str(v) + '_' + shape1 + '.jpg')
            except:
                try:
                    img = Image.open(train_folder_name+'/' + str(v) + '_' + shape2 + '.jpg')
                except:
                    img = Image.open(train_folder_name+'/' + str(v) + '_' + shape3 + '.jpg')
            im_matrix = np.array(img)
            im_matrix = im_matrix.flatten()
            im_matrix = im_matrix.astype('float64')
            globals()["im_matrix" + str(v)] = im_matrix / 255
            xminus.append(globals()["im_matrix" + str(v)])
            x = x + globals()["im_matrix" + str(v)]
            nonhearts = nonhearts + 1
        except:
            pass
    mminus = x / nonhearts

    r = np.linalg.norm(mplus - mminus)
    xvector = []
    checktype = []
    for v in range(1, val + 1):

        try:
            img = Image.open(train_folder_name+'/' + str(v) + '_C.jpg')
            checktype.append("C")
        except:
            try:
                img = Image.open(train_folder_name+'/' + str(v) + '_S.jpg')
                checktype.append("S")
            except:
                try:
                    img = Image.open(train_folder_name+'/'+ str(v) + '_D.jpg')
                    checktype.append("D")
                except:
                    img = Image.open(train_folder_name+'/' + str(v) + '_H.jpg')
                    checktype.append("H")

        im_matrix = np.array(img)
        im_matrix = im_matrix.flatten()
        im_matrix = im_matrix.astype('float64')
        globals()["im_matrix" + str(v)] = im_matrix / 255
        xvector.append(globals()["im_matrix" + str(v)])
    rplusvector = []
    for i in xvector:
        newval = np.linalg.norm(i - mplus)
        rplusvector.append(newval)
    rplus = max(rplusvector)

    rminusvector = []
    for i in xvector:
        newval = np.linalg.norm(i - mminus)
        rminusvector.append(newval)
    rminus = max(rminusvector)

    lamda = r / (rplus + rminus)

    l = 0
    for i in xvector:
        l = l + i
    mcentroid = l / val

    xiprime = []
    for i in xvector:
        cv= i*lamda+(1-lamda)*mcentroid
        xiprime.append(cv)

    neeewvaal = shape

    if neeewvaal  == shape:
        lop = -1
        for v in range(1,val+1):
            try:
                lop = lop+1
                img = Image.open(train_folder_name+'/'+str(v)+'_'+shape+'.jpg')
                im_matrix = np.array(img)
                im_matrix = im_matrix.flatten()
                im_matrix = im_matrix.astype('float64')
                globals()["positive" + str(lop)] = im_matrix/255
                break
            except:
                pass
        vop = -1
        alfa = 0
        for v in range(1,val+1):
            vop = vop +1
            try:
                img = Image.open(train_folder_name+'/'+str(v)+'_'+shape1+'.jpg')
                alfa = 1
            except:
                try:
                    img = Image.open(train_folder_name+'/'+str(v)+'_'+shape2+'.jpg')
                    alfa = 1
                except:
                    try:
                        img = Image.open(train_folder_name+'/'+str(v)+'_'+shape3+'.jpg')
                        alfa = 1
                    except:
                        pass
            if alfa == 1:
                im_matrix = np.array(img)
                im_matrix = im_matrix.flatten()
                im_matrix = im_matrix.astype('float64')
                globals()["nonpositive" + str(vop)] = im_matrix/255
                break
    A = np.dot(globals()["positive" + str(lop)],globals()["positive" + str(lop)])+1
    B = np.dot(globals()["nonpositive" + str(vop)],globals()["nonpositive" + str(vop)])+1
    C = np.dot(globals()["positive" + str(lop)],globals()["nonpositive" + str(vop)])+1
    A = pow(A,4)
    B = pow(B,4)
    C = pow(C,4)
    D = []
    for y in xiprime:
        gk = np.dot(y,globals()["positive" + str(lop)])+1
        gk = pow(gk,4)
        D.append(gk)
    E = []
    for y in xiprime:
        gk = np.dot(y,globals()["nonpositive" + str(vop)])+1
        gk = pow(gk,4)
        E.append(gk)
    import math
    kk = -1
    mi = []
    firstval = 0
    nonval = 0
    for i in D:
        kk = kk+1
        if checktype[kk] == shape:
            lklk = A+B-(2*C)
            ghgh = (i-E[kk]+B-C)/math.sqrt(lklk)
            mi.append(ghgh)
        else:
            lklk = A+B-(2*C)
            ghgh = (E[kk]-i+A-C)/math.sqrt(lklk)
            mi.append(ghgh)
    kall = np.argmin(mi, axis = 0)
    kal = mi[kall]
    checktypee = []
    for jjj in checktype:
        checktypee.append(jjj)
    jj = -1
    for ghh in checktypee:
        jj = jj+1
        if ghh == shape:
            if firstval == 0:
                checktypee[jj] = 1
                firstval = firstval+1
            else:
                checktypee[jj] = 0
        elif nonval == 0:
                checktypee[jj] = 1
                nonval = nonval+1
        else:
            checktypee[jj] = 0

    epsilon = epsilon
    avector = []

    ytyt = 0
    while True:
        ytyt = ytyt + 1
        if ytyt % max_updates == 0:
            break

        if math.sqrt(lklk) - kal < epsilon:
            break
        else:
            if checktype[kall] == shape:
                K = np.dot(xiprime[kall], xiprime[kall]) + 1
                K = pow(K, 4)
                adapt = (A - D[kall] + E[kall] - C) / (A + K - 2 * (D[kall] - E[kall]))
                q = min(1, adapt)
                if q == 1:

                    break
                A = A * pow((1 - q), 2) + (2 * (1 - q) * q * D[kall]) + (pow(q, 2) * K)
                C = (1 - q) * C + (q * E[kall])
                kl = -1
                for i in D:
                    kl = kl + 1
                    KK = np.dot(xiprime[kl], xiprime[kall]) + 1
                    KK = pow(KK, 4)
                    vallll = (1 - q) * i + (q * KK)
                    D[kl] = vallll
                fgfg = -1
                for i in checktypee:
                    fgfg = fgfg + 1
                    if fgfg == kall:
                        typeee = (1 - q) * i + q * 1
                    else:
                        typeee = (1 - q) * i + (q * 0)
                    checktypee[fgfg] = typeee
                import math

                kk = -1
                mi = []
                firstval = 0
                nonval = 0

                for i in D:
                    kk = kk + 1
                    if checktype[kk] == shape:
                        lklk = A + B - (2 * C)
                        ghgh = (i - E[kk] + B - C) / math.sqrt(lklk)
                        mi.append(ghgh)
                    else:
                        lklk = A + B - (2 * C)
                        ghgh = (E[kk] - i + A - C) / math.sqrt(lklk)
                        mi.append(ghgh)
                kall = np.argmin(mi, axis=0)

                kal = mi[kall]
                epsilon = epsilon


            else:
                K = np.dot(xiprime[kall], xiprime[kall]) + 1
                K = pow(K, 4)
                adapt = (B - E[kall] + D[kall] - C) / (B + K - 2 * (E[kall] - D[kall]))
                q = min(1, adapt)
                if q == 1:

                    break
                B = B * pow((1 - q), 2) + (2 * (1 - q) * q * E[kall]) + (pow(q, 2) * K)
                C = ((1 - q) * C + q * D[kall])
                kl = -1
                for i in E:
                    kl = kl + 1
                    KK = np.dot(xiprime[kl], xiprime[kall]) + 1
                    KK = pow(KK, 4)
                    vallll = (1 - q) * i + (q * KK)
                    E[kl] = vallll
                fgfg = -1
                for i in checktypee:
                    fgfg = fgfg + 1
                    if fgfg == kall:
                        typeee = (1 - q) * i + q * 1
                    else:
                        typeee = (1 - q) * i + (q * 0)
                    checktypee[fgfg] = typeee
                import math

                kk = -1
                mi = []
                firstval = 0
                nonval = 0

                for i in D:
                    kk = kk + 1
                    if checktype[kk] == shape:
                        lklk = A + B - (2 * C)
                        ghgh = (i - E[kk] + B - C) / math.sqrt(lklk)
                        mi.append(ghgh)
                    else:
                        lklk = A + B - (2 * C)
                        ghgh = (E[kk] - i + A - C) / math.sqrt(lklk)
                        mi.append(ghgh)
                kall = np.argmin(mi, axis=0)

                kal = mi[kall]
                epsilon = epsilon
    yvector = []
    llll = -1
    for i in checktypee:
        llll = llll+1
        if i != 0:
            if checktype[llll] == shape:
                yvector.append(1)
            else:
                yvector.append(-1)
    checktypeenew = []
    checktypeenew1 = []
    jj = -1
    for i in checktypee:
        jj = jj+1
        if i != 0:
            checktypeenew.append([i,jj])
            checktypeenew1.append([jj,i])

    # ltt = np.dot(checktypeenew,yvector)
    xvector = []
    checktype = []
    for v in range(1, val + 1):

        try:
            img = Image.open(train_folder_name+'/' + str(v) + '_C.jpg')
            checktype.append("C")
        except:
            try:
                img = Image.open(train_folder_name+'/' + str(v) + '_S.jpg')
                checktype.append("S")
            except:
                try:
                    img = Image.open(train_folder_name+'/' + str(v) + '_D.jpg')
                    checktype.append("D")
                except:
                    img = Image.open(train_folder_name+'/' + str(v) + '_H.jpg')
                    checktype.append("H")

        im_matrix = np.array(img)
        im_matrix = im_matrix.flatten()
        im_matrix = im_matrix.astype('float64')
        globals()["im_matrix" + str(v)] = im_matrix / 255
        xvector.append(globals()["im_matrix" + str(v)])

    import pickle

    data = {"mplus":mplus,"mminus":mminus,"lamda":lamda,"A":A,"B":B,"weights":checktypeenew1}

    filename = model_file_name

    if model_file_name.lower().find("clu") != -1:
        if shape== "C":
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if model_file_name.lower().find("hea") != -1:
        if shape== "H":
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if model_file_name.lower().find("spad") != -1:
        if shape== "S":
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if model_file_name.lower().find("diam") != -1:
        if shape== "D":
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)




i = 0
for arg in sys.argv:

	globals()["arg" + str(i)] = arg
	i = i+1

epsilon  = float(arg1)
max_updates  = int(arg2)
class_letter = arg3
model_file_name = arg4
train_folder_name = arg5

svmviewer(epsilon = epsilon,max_updates = max_updates,class_letter = class_letter,model_file_name  = model_file_name,train_folder_name = train_folder_name)


