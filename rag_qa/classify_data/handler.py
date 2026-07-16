#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_json("training_data_1000.json", lines=False)
print(data.head())
data.to_json("training_data_1000.json", orient="records", lines=True, index=None, force_ascii=False)
