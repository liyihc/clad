Codes of the article [A Contrastive Learning Framework for Detecting Anomalous Behavior in Commodity Trading Platforms](https://www.mdpi.com/2076-3417/13/9/5709)

You can use the `requirements.txt` file or can use packages with the same major version number.

# Usage

- download this repository
- prepare your python environment with pytorch cuda support. (I recommend to use conda)
- generate config file.
  ```bash
  # python clad-cli.py get-config -h
  python clad-cli.py get-config eshop gru-all > a.json
  ```
- modify config file
- run model, and the result will be in the `output` folder.
  ```bash
  # python clad-cli.py run -h
  python clad-cli.py run a.json
  ```
