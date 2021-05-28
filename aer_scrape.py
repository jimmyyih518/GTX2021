# -*- coding: utf-8 -*-
"""
Spyder Editor

User: JZ
Created: 2021-05-27
"""

import pandas as pd
from copy import copy

def uwi_convert(uwi):
    uwi_tolist = [char for char in uwi]
    if len(uwi_tolist) != 16:
        print('not 16 digit uwi, breaking...')
        return None
    else:
        uwi_tolist = uwi_tolist[1:-1]
        uwi_fmt = uwi_tolist[0:2] + ['/'] + uwi_tolist[2:4] + ['-'] + uwi_tolist[4:6] + ['-'] + uwi_tolist[6:9] + ['-'] + uwi_tolist[9:13] + ['/'] + [uwi_tolist[13]]
        uwi_fmt = ''.join(uwi_fmt)
        return uwi_fmt

def aer_scrape(uwi, tablename):
    uwi = uwi_convert(uwi)
    url = f'https://www2.aer.ca/t/Production/views/{tablename}.csv?Enter%20Well%20Identifier%20(UWI)={uwi}'
    df = pd.read_csv(url)
    return(df)

def aer_scrape_multiple(uwi_list, table_list):
    output = {}
    
    for tablename in table_list:
        print('scraping ' + tablename)
        if 'combine_df' in locals() or 'combine_df' in globals():
            del combine_df
        for uwi in uwi_list:
            print('getting data for ' + uwi)
            try:
                df = aer_scrape(uwi, table_list[tablename])
                if 'combine_df' in locals() or 'combine_df' in globals():
                    combine_df = combine_df.append(df, ignore_index = True)
                else:
                    combine_df = copy(df)
            except Exception:
                pass

        output[tablename] = combine_df
        
    return output
    
#df_dict = aer_scrape_multiple(uwi, aer_tables)

# aer_tables = {'Header':'PRD_0100_Well_Summary_Report_Download/WellInformation',
#               'CoreAnalysis':'PRD_0100_Well_Summary_Report_Download/CoreAnalysisDetail',
#               'CoreAnalysisDetail':'PRD_0100_Well_Summary_Report_Download/CoreAnalysisLineDetail',
#               'GeoTops':'PRD_0100_Well_Summary_Report_Download/GeologicalTopsMarkers',
#               'GasProps':'0125_Well_Gas_Analysis_Data_EXT_Detail_Download/WellGas-Properties',
#               'GasHeader':'0125_Well_Gas_Analysis_Data_EXT_Detail_Download/WellGas-Header',
#               'SepLiqComp':'0125_Well_Gas_Analysis_Data_EXT_Detail_Download/WellGas-SeparatorFluidData',
#               'ResGasComp':'0125_Well_Gas_Analysis_Data_EXT_Detail_Download/WellGas-ReservoirData'}

# uwi = ['100061706804W600','100071406220W500']
# uwi = ['100061706804W600','100072507623W500']
# df = aer_scrape_multiple(uwi, aer_tables)

