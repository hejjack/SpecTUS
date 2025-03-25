The data were created from the original NIST slit with code in notebooks/2_forward_models_training.ipynb
It was first transformed into an .sdf format and then curated in a way NEIMS supports. For details on
the curation process, see the notebook.

Final calls:
```python
# run the transformation
transform_to_neims_sdf_format(nist_train_path.replace(".msp", ".sdf"),
                              output_train)

transform_to_neims_sdf_format(nist_test_path.replace(".msp", ".sdf"),
                              output_test)

transform_to_neims_sdf_format(nist_valid_path.replace(".msp", ".sdf"),
                              output_valid)
```