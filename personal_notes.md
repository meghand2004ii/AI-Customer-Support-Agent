## Model Interpretation, Persistence, and Retention List

The Random Forest model identified frequency as the most important feature (0.3281), followed by monetary value (0.2721), recency (0.1007), cancellation rate (0.0541), and the Electronics category (0.0509). These results indicate that customer purchasing behaviour is highly informative for predicting lapse.

The Logistic Regression model showed a negative coefficient for frequency (-1.2811), suggesting that customers with higher purchase frequency have lower predicted lapse odds, holding other variables constant. The Electronics (-0.3782) and Snacks (-0.2026) category coefficients were also negative relative to their respective reference category.

The selected Logistic Regression pipeline was saved as `outputs/lapse_model.joblib`. After reloading the model, predictions on the test set matched the original predictions exactly. A ranked Premium retention list was generated and saved as `outputs/retention_list.csv`. Among the 10 highest-ranked Premium customers in the test set, 2 were actually lapsed.

### Limitations and Next Steps

The Premium retention list is illustrative rather than a leakage-free prospective evaluation because the Premium segment flag is derived from the broader RFM analysis, which includes activity beyond the September lapse-label window. The lapse prediction model itself uses only pre-cutoff features. Preprocessing was fitted within the training pipeline, and the test set was not used to fit the encoders, scalers, or model. Oversampling was performed only on the training data. No Task 3 segment labels were used as model features, and no data from September 1, 2025 onward was included in the prediction features.

Future work could evaluate the model on a later, fully held-out time period and test retention campaigns using an appropriate randomized experiment.