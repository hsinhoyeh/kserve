# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import asyncio
import base64
import io
import logging
import kfserving
import numpy as np
from aix360.algorithms.lime import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PIL import Image


class AIXModel(kfserving.KFModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, predictor_host: str, segm_alg: str, num_samples: str,
                 top_labels: str, min_weight: str, positive_only: str, explainer_type: str):
        super().__init__(name)
        self.name = name
        self.top_labels = int(top_labels)
        self.num_samples = int(num_samples)
        self.segmentation_alg = segm_alg
        self.predictor_host = predictor_host
        self.min_weight = float(min_weight)
        self.positive_only = (positive_only.lower() == "true") | (positive_only.lower() == "t")
        if str.lower(explainer_type) != "limeimages":
            raise Exception("Invalid explainer type: %s" % explainer_type)
        self.explainer_type = explainer_type
        self.ready = False

    def load(self) -> bool:
        self.ready = True
        return self.ready

    def _predict(self, input_im):
        #scoring_data = {'instances': input_im.tolist()}
        scoring_data = self._wrap_numpyarr_to_predict_inputs(input_im)

        loop = asyncio.get_running_loop()
        resp = loop.run_until_complete(self.predict(scoring_data))
        predictions = resp["predictions"]
        logging.info("_predict.results:%s", predictions)
        # output: "predictions": [
        #        {
        #            "scores": [1.47944235e-07, 3.65586068e-08, 0.796582818, 1.05895253e-07, 0.203416958, 3.8090274e-08],
        #            "prediction": 2,
        #            "key": "1"
        #        }
        #    ]
        #}
        num_classes = len(predictions[0]["scores"])
        num_samples = len(predictions)
        ## class_preds is in such shape:
        ## [ [sample_1_to_class_1, sample_1_to_class_2, ...] [sample_2_to_class_1, ], ... ]
        class_preds = [[] for x in range(0, num_samples)]
        for sample_index in range(0, num_samples):
                class_preds[sample_index] = predictions[sample_index]["scores"]
        logging.info("_predict.classpred:%s", class_preds)
        return np.array(class_preds)

    def _wrap_numpyarr_to_predict_inputs(self, input_im: np.ndarray):
        logging.info("numpyarr wrapper:%s",type(input_im))
        logging.info("numpyarr wrapper:%s",input_im.shape)
        instances = []
        index = 1;
        for slice_input_im in input_im:
            pil_img = Image.fromarray(slice_input_im)
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            input_base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")    
            instances.append({"image_bytes": {"b64": input_base64_str}, "key": "{}".format(index)})
            index = index + 1

        return {"instances": instances}

    def _get_instance_binary_inputs(self, first_instance):
        logging.info("first instance type:%s",type(first_instance))
        logging.info("first instance isdict:%s",isinstance(first_instance, dict))
        logging.info("first instance hasb64:%s", "b64" in first_instance)
        logging.info("first instance keys:%s", first_instance.keys())

        if isinstance(first_instance, dict) and "image_bytes" in first_instance and "b64" in first_instance["image_bytes"]: # first_instance = {"image_bytes": {"b64":xxx}}
            logging.info("first instance is dict and has b64, coverting")
            return Image.open(io.BytesIO(base64.b64decode(first_instance["image_bytes"]["b64"])))
        return first_instance

    def explain(self, request: Dict) -> Dict:
        instances = request["instances"]
        try:
            top_labels = (int(request["top_labels"]) 
                          if "top_labels" in request else 
                          self.top_labels)
            segmentation_alg = (request["segmentation_alg"] 
                                if "segmentation_alg" in request else 
                                self.segmentation_alg)
            num_samples = (int(request["num_samples"]) 
                           if "num_samples" in request else 
                           self.num_samples)
            positive_only = ((request["positive_only"].lower() == "true") | (request["positive_only"].lower() == "t") 
                             if "positive_only" in request else 
                             self.positive_only)
            min_weight = (float(request['min_weight']) 
                          if "min_weight" in request else 
                          self.min_weight)
        except Exception as err:
            raise Exception("Failed to specify parameters: %s", (err,))

        try:
            #inputs = np.array(instances[0])
            inputs = np.array(self._get_instance_binary_inputs(instances[0]))
            logging.info("Calling explain on image of shape %s", (inputs.shape,))
        except Exception as err:
            raise Exception(
                "Failed to initialize NumPy array from inputs: %s, %s" % (err, instances))
        try:
            if str.lower(self.explainer_type) == "limeimages":
                explainer = LimeImageExplainer(verbose=False)
                segmenter = SegmentationAlgorithm(segmentation_alg, kernel_size=1,
                                                  max_dist=200, ratio=0.2)
                explanation = explainer.explain_instance(inputs,
                                                         classifier_fn=self._predict,
                                                         top_labels=top_labels,
                                                         hide_color=0,
                                                         num_samples=num_samples,
                                                         segmentation_fn=segmenter)

                temp = []
                masks = []
                for i in range(0, top_labels):
                    temp, mask = explanation.get_image_and_mask(explanation.top_labels[i],
                                                                positive_only=positive_only,
                                                                num_features=10,
                                                                hide_rest=False,
                                                                min_weight=min_weight)
                    masks.append(mask.tolist())

                return {"explanations": {
                    "temp": temp.tolist(),
                    "masks": masks,
                    "top_labels": np.array(explanation.top_labels).astype(np.int32).tolist()
                }}

        except Exception as err:
            raise Exception("Failed to explain %s" % err)
