# Data Dictionary

| Field | Type | Role |
|---|---|---|
| application_date | date | temporal split |
| credit_score | float/null | traditional model feature |
| credit_history_years | float | traditional |
| on_time_ratio | float | repayment |
| delinquencies_24m | int | repayment |
| months_since_last_delinquency | float | repayment |
| utility_rent_regularity | float | alternative |
| income_volatility | float | alternative cash flow |
| avg_monthly_inflow | float | alternative cash flow |
| overdraft_count | int | alternative cash flow |
| savings_buffer_months | float | alternative cash flow |
| gig_income_share | float | alternative cash flow |
| employment_tenure_years | float | alternative |
| dti | float | traditional |
| requested_amount | float | traditional/application |
| term_months | int | traditional/application |
| thin_file/no_file | int | segmentation/control variables |
| default_12m | int | synthetic target |
| age_band/sex/race_ethnicity/zip_income_tier | categorical | fairness audit only; never model features |
