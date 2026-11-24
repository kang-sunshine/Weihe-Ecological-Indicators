# -*- coding: utf-8 -*-
"""Report optional120d coverage only when Extreme support is spatially sparse."""
import json
import numpy as np
import pandas as pd
from core import *
def main():
 if (O/'qc/FIXED120_REPORT_GATE_READY.json').exists():return
 d=pd.read_csv(O/'outputs/C2_FixedWindow_Legacy.csv');bins=pd.read_csv(O/'outputs/C3_SeverityLegacy_Sensitivity.csv');support_rows=[]
 for h in HAZ:
  ext=d[d.Hazard.eq(h)&d.Window_Days.eq(120)&d.Row_Type.eq('GROUP_L')&d.Severity_Group.eq('Extreme')&d.Statistic.eq('Mean')].iloc[0];ok=ext.N_events>=500 and ext.N_50km_blocks>=20
  support_rows.append(dict(Hazard=h,Window_Days=120,N_Extreme_Events=int(ext.N_events),N_Extreme_50km_Blocks=int(ext.N_50km_blocks),Minimum_Events=500,Minimum_Spatial_Blocks=20,Report_L_Contrasts=bool(ok),Decision_Basis='uniformsupport-onlyqualitygate,notdirection/significance; optional120notrequiredcorewindow'))
  if ok:continue
  mask=d.Hazard.eq(h)&d.Window_Days.eq(120)&d.Row_Type.isin(['GROUP_L','DIRECT_SPATIAL_CONTRAST'])
  for col in ['Estimate','CI_Lower','CI_Upper']:d.loc[mask,col]=np.nan
  d.loc[mask,'CI_Status']='NOT ESTIMABLE_FOR_REPORTING_SPATIAL_SUPPORT';d.loc[mask,'Missing_Reason']='optional120Extreme<20independent50kmblocks; coverageonly; do not interpretgroupLdifference'
  bm=bins.Hazard.eq(h)&bins.Window_Days.eq(120)
  for col in ['Original_L_Median','Fixed_L_Median']:bins.loc[bm,col]=np.nan
  bins.loc[bm,'Status']='COVERAGE_ONLY_120_SPATIAL_SUPPORT';bins.loc[bm,'Missing_Reason']='optional120Extreme<20blocks; responsecurvewithheld'
 d.to_csv(O/'outputs/C2_FixedWindow_Legacy.csv',index=False,encoding='utf-8-sig');bins.to_csv(O/'outputs/C3_SeverityLegacy_Sensitivity.csv',index=False,encoding='utf-8-sig');pd.DataFrame(support_rows).to_csv(O/'qc/Fixed120_Reporting_Support.csv',index=False)
 reps=pd.read_csv(O/'qc/C2_Spatial_Contrast_Replicates.csv.gz');reps['Reporting_Eligible']=reps.Window_Days<120;reps['Note']=np.where(reps.Reporting_Eligible,'corefixedwindows','computedQC_only; optional120supportgatewithholdsinterpretation');reps.to_csv(O/'qc/C2_Spatial_Contrast_Replicates.csv.gz',index=False,compression='gzip')
 (O/'qc/FIXED120_REPORT_GATE_READY.json').write_text(json.dumps(dict(Completed=True,Gate='ExtremeN>=500 andoccupied50kmblocks>=20; otherwise optional120coverageonly',Computed_Replicates_Retained_As_Nonreportable_QC=True)),encoding='utf8')
 print('OPTIONAL120 SUPPORT GATE',support_rows,flush=True)
if __name__=='__main__':main()
