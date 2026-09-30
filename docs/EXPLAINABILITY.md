# Explainability and Reason Codes

For the alternative-data GBDT, SHAP TreeExplainer identifies the largest absolute local contributions. Those features are mapped through `config/reason_codes.yaml` to stable human-readable reason codes. Only factors actually present in the top contribution set can generate reasons; there is no generic canned decline list.

The borrower message is template-generated with Jinja2 and contains no LLM. It is a prototype of an ECOA/Reg B-style adverse-action explanation pattern. A real lender must have counsel/compliance validate the reason-code taxonomy, timing, specificity, consistency with the actual model, and interaction with applicable federal/state requirements before use.
