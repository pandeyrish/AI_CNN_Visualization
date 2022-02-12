import sys
def testmesvm(model_file_name ,train_folder_name,test_folder_name):
    import pickle
    import os
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter


    val = len(os.listdir(train_folder_name))
    if val < 1:
        notrainingdata = "NO TRAINING DATA"
        return print(notrainingdata)

    val1 = len(os.listdir(test_folder_name))

    if val1 < 1:
        notestdata = "NO TEST DATA"
        return print(notestdata)

    try:
        with open(model_file_name, 'rb') as handle:
            b = pickle.load(handle)
    except:
        modelfilewrong = "MODEL FILE IS NOT OF THE CORRECT FORMAT"
        return print(modelfilewrong)
    if model_file_name.lower().find("clu") != -1:
        shape = "C"

    if model_file_name.lower().find("hea") != -1:
        shape = "H"

    if model_file_name.lower().find("spad") != -1:
        shape = "S"

    if model_file_name.lower().find("diam") != -1:
        shape = "D"

    testchecktype = []

    for v in range(1, val1 + 1):

        try:
            img = Image.open(test_folder_name + '/' + str(v) + '_C.jpg')
            testchecktype.append("C")
        except:
            try:
                img = Image.open(test_folder_name + '/' + str(v) + '_S.jpg')
                testchecktype.append("S")
            except:
                try:
                    img = Image.open(test_folder_name + '/' + str(v) + '_D.jpg')
                    testchecktype.append("D")
                except:
                    img = Image.open(test_folder_name + '/' + str(v) + '_H.jpg')
                    testchecktype.append("H")

    xtestvector = []
    for v in range(1, val1 + 1):

        try:
            img = Image.open(test_folder_name+'/' + str(v) + '_C.jpg')

        except:
            try:
                img = Image.open(test_folder_name+'/' + str(v) + '_S.jpg')

            except:
                try:
                    img = Image.open(test_folder_name+'/' + str(v) + '_D.jpg')

                except:
                    img = Image.open(test_folder_name+'/' + str(v) + '_H.jpg')


        im_matrix = np.array(img)
        im_matrix = im_matrix.flatten()
        im_matrix = im_matrix.astype('float64')
        globals()["im_matrix" + str(v)] = im_matrix / 255
        xtestvector.append(globals()["im_matrix" + str(v)])
    print(len(xtestvector))
    A = float(b["A"])
    B = float(b["B"])
    lamda = float(b["lamda"])
    mplus = b["mplus"]
    mminus = b["mminus"]
    checktypeenew = b["weights"]
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

    l = 0
    for i in xvector:
        l = l+i
    mcentroid = l/val

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

    yvector = []
    llll = -1
    for i in checktypeenew:

        llll = llll+1

        if checktype[llll] == shape:
            yvector.append(1)
        else:
            yvector.append(-1)


    xiprime = []
    for i in xvector:
        cv= i*lamda+(1-lamda)*mcentroid
        xiprime.append(cv)

    lll = 0
    llll = -1
    correct = 0
    falsepositive = 0
    falsenegative = 0
    print(len(xtestvector))
    for j in xtestvector:
        lds = -1
        submission = 0
        for i in checktypeenew:
            lds = lds+1
            K = np.dot(j,xiprime[i[0]])+1
            K = pow(K,4)
            submission = submission+i[1]*yvector[lds]*K
        submission = submission+((B-A)/2)
        if submission > 0:
            llll = llll + 1
            lll = lll+1
            if testchecktype[llll] == shape:
                print(lll," ","Correct")
                correct = correct + 1
            else:
                print(lll, " ", "False Positive")
                falsepositive = falsepositive + 1
        else:
            llll = llll + 1
            lll = lll + 1
            if testchecktype[llll] == shape:
                print(lll," ","False Negative")
                falsenegative = falsenegative + 1
            else:
                print(lll, " ", "Correct")
                correct = correct + 1
    fractioncorrect = correct/val1

    fractionfalsepostive = falsepositive/val1

    fractionfalsenegative = falsenegative/val1

    print("Fraction Correct:"," ",fractioncorrect)
    print("Fraction False Positive:", " ", fractionfalsepostive)
    print("Fraction False Negative:", " ", fractionfalsenegative)



i = 0
for arg in sys.argv:

	globals()["arg" + str(i)] = arg
	i = i+1

model_file_name = arg1
train_folder_name  = arg2
test_folder_name = arg3


testmesvm(model_file_name =model_file_name,train_folder_name = train_folder_name,test_folder_name = test_folder_name)


