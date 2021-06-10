# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:54:44 2021

@author: JZ2018
"""

from aer_scrape import *
uwi_dvn_tt = pd.read_csv('dvn_tt_uwi.csv', header=None)
uwi_dvn_tt.columns=['UWI']
aer_tables = {'Header':'PRD_0100_Well_Summary_Report_Download/WellInformation'}
aer_download = aer_scrape_multiple(uwi_dvn_tt['UWI'], aer_tables)
aer_download['Header'].to_csv('data/Dvn_TT_AER.csv')
