from datetime import datetime
import numpy as np
import cv2
import xlsxwriter
import glob
import pandas as pd
import matplotlib.pyplot as plt


# from https://github.com/microsoft/HoloLens2ForCV/
def load_lut(lut_filename):
    with open(lut_filename, mode='rb') as depth_file:
        lut = np.frombuffer(depth_file.read(), dtype="f")
        lut = np.reshape(lut, (-1, 3))
    return lut

# from https://github.com/microsoft/HoloLens2ForCV/
def get_points_in_cam_space(img, lut):
    img = np.tile(img.flatten().reshape((-1, 1)), (1, 3))
    points = img * lut
    remove_ids = np.where(np.sum(points, axis=1) < 1e-6)[0]
    points = np.delete(points, remove_ids, axis=0)
    points /= 1000.
    return points

TASK = "graph"

if TASK == "graph":
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    data = pd.read_excel('./mat.xlsx', header=None)

    for i in range(0, data.shape[0]):
        data[0][i] = data[0][i] + 2
        data[1][i] = data[1][i] + 2

    CELLCOUNT = 10 # number of bins lengths per meter
    rows, cols = (8*CELLCOUNT, 8*CELLCOUNT)
    arr1 = []
    arr2 = []
    arr3 = []
    for i in range(rows):
        col1 = []
        col2 = []
        col3 = []
        for j in range(cols):
            col1.append(0)
            col2.append(0)
            col3.append(0)
        arr1.append(col1)
        arr2.append(col2)
        arr3.append(col3)

    for i in range(0, len(data[0])):
        row = round(data[0][i]*CELLCOUNT)
        col = round(data[1][i]*CELLCOUNT)
        z = data[2][i]
        if z < -0.5:
            arr1[row][col] += 1
        elif z < 0.5:
            arr2[row][col] += 1
        else:
            arr3[row][col] += 1

    xdata = []
    ydata = []
    cdata = []
    zdata = []
    for i in range(0, len(arr1)):
        for j in range(0, len(arr1[i])):
            xdata.append(i)
            ydata.append(j)
            zdata.append(0)
            cdata.append(arr1[i][j])
    for i in range(0, len(arr2)):
        for j in range(0, len(arr2[i])):
            xdata.append(i)
            ydata.append(j)
            zdata.append(1)
            cdata.append(arr2[i][j])
    for i in range(0, len(arr3)):
        for j in range(0, len(arr3[i])):
            xdata.append(i)
            ydata.append(j)
            zdata.append(2)
            cdata.append(arr3[i][j])

    OCCUPANCY_GRID_MAP = True
    if OCCUPANCY_GRID_MAP:
        ax.scatter3D(xdata, ydata, zdata, c=cdata, cmap='gray', s=0.6)
        plt.show()
    else:
        maxc = max(cdata)
        for c in range(0, len(cdata)):
            if cdata[c] > maxc/5:
                cdata[c] = maxc
            else:
                cdata[c] = 0
        ax.scatter3D(xdata, ydata, zdata, c=cdata, cmap='gray', s=0.6)
        plt.show()

if TASK == "time":
    # This is the raw data
    filetimes_raw = "132809390074102586 132809390084118183 132809390094116628 132809390104116829 132809390114117031 132809390124117228 132809390134117427 132809390144117625 132809390154133163 132809390164131642 132809390174131838 132809390184132034 132809390194132227 132809390204132421 132809390214132614 132809390224149992 132809390234148465 132809390244148655 132809390254148845 132809390264149033 132809390274149222 132809390284149411 132809390294164875 132809390304163320 132809390314163512 132809390324163706 132809390334163899 132809390344164091 132809390354164284 132809390364179818 132809390374178289 132809390384178478 132809390394178671 132809390404178860 132809390414179050 132809390424179238 132809390434179427 132809390444179615 132809390454179805 132809390464179993 132809390474180182 132809390484196440"
    filetimes = filetimes_raw.split(" ")
    # Go through the data and print the datetime
    for f in filetimes:
        f = int(f)
        print(datetime.utcfromtimestamp((f - 116444736000000000) / 10000000))

if TASK == "data":
    # Go through each folder that starts with data
    bin_file = glob.glob("./*_lut.bin")
    bin_file = bin_file[0]
    lut = load_lut(bin_file)
    pngs = glob.glob("./pngs/*")
    pngs = sorted(pngs)
    workbook = xlsxwriter.Workbook('./output/out.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for png in pngs:
        print("processing " + png[7:])

        # Load the files and get the points
        im = cv2.imread(png, -1)
        pts = get_points_in_cam_space(im, lut)
        # Print the points to the xlsx file

        for col, data in enumerate(pts):
            worksheet.write_column(row, col, data)
        row = row + 3
    workbook.close()
