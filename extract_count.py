# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:39:28 2018

@author: joach
"""

#import libraries
import xlrd
import xlwt

#Initiate write file
wt_workbook = xlwt.Workbook()
wt_sheet = wt_workbook.add_sheet("tellingen_totaal")

#Add headers
wt_sheet.write(0,0, 'telpuntnr')
wt_sheet.write(0,1, 'locatie')
wt_sheet.write(0,2, 'richting1')
wt_sheet.write(0,3, 'r1_intens')
wt_sheet.write(0,4, 'richting2')
wt_sheet.write(0,5, 'r2_intens')
wt_sheet.write(0,6, 'lat')
wt_sheet.write(0,7, 'long')

#Initiate read file
file_location = "C:/Thesis/Data/Tellingen/tellingen_tilburg_2015.xls"
rd_workbook = xlrd.open_workbook(file_location)
row = 1
col = 0

#Loop through sheets
for rd_sheet in rd_workbook.sheets():
     wt_sheet.write(row, col + 0, rd_sheet.cell_value(4, 1))
     wt_sheet.write(row, col + 1, rd_sheet.cell_value(6, 1))
     wt_sheet.write(row, col + 2, rd_sheet.cell_value(8, 1))
     wt_sheet.write(row, col + 3, rd_sheet.cell_value(35, 1))
     wt_sheet.write(row, col + 4, rd_sheet.cell_value(10, 1))
     wt_sheet.write(row, col + 5, rd_sheet.cell_value(35, 4))
     wt_sheet.write(row, col + 6, rd_sheet.cell_value(9, 1))
     wt_sheet.write(row, col + 7, rd_sheet.cell_value(9, 2))
     row += 1
    
#Save workbook to new xls
wt_workbook.save("tellingen_compleet.xls") 