from elg import Corpus, Service
import zipfile
import os
import csv
from glob import glob
from xml.etree import ElementTree
from sklearn.metrics import classification_report
import logging as log

log.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=log.INFO
    )

DATA_DIR = "./data"
    
tasks = {
    "haspeede": {
        "id": 7498,
        "task": "classification"
    },
    "textualentailment": {
        "id": 8121,
        "task": "pairs"
    },
    "eventi": {
        "id": 7376,
        "task": "sequence"
    },
    "sentipolc": {
        "id": 7479,
        "task": "classification"
    },
    "facta": {
        "id": 8045,
        "task": "sequence"
    },
    "ironita": {
        "id": 7372,
        "task": "classification"
    }
}

class DataReader():

    def __init__(self, _task_name: str, _data_dir: str = DATA_DIR):
        self.task_name = _task_name
        self.training_set = []
        self.test_set = []
        self._data_dir = _data_dir
        self._task_data_dir = os.path.join(self._data_dir, self.task_name)
        self._task_data_zip = os.path.join(self._data_dir, f"{self.task_name}.zip")
        self._task_data_desc = os.path.join(self._data_dir, f"{self.task_name}.txt")

        if not os.path.isdir(_data_dir):
            os.mkdir(_data_dir)
        
        self.target_values = []
        self.target_desc = []
        self.feature_info = []
        
        self._read_data()
        
        self.keys = list(self.training_set[0].keys())
        self.target_key = 'labels' if self.task_name in ["eventi", "facta"] else 'label'
        self.target_dim = 'multiple' if self.task_name in ["eventi", "facta"] else 'single'
        self.feature_keys = [x for x in self.keys if x not in ['id', self.target_key]]
        self.feature_dim = ['multiple'] if self.task_name in ["eventi", "facta"] else ['single']*2 if self.task_name == 'textualentailment' else ['single']
        
    def _download_data(self):
        if os.path.isfile(self._task_data_zip):
            log.info(f"zip file already downloaded: {self._task_data_zip}")
        else:
            corpus = Corpus.from_id(tasks[self.task_name]["id"])
            # store description
            with open(self._task_data_desc, 'w') as f:
                f.write(str(corpus))
            corpus.download(filename=self.task_name, folder=self._data_dir)
        
    def _unzip_data(self):

        if os.path.isdir(self._task_data_dir) and len(os.listdir(self._task_data_dir)) > 0:
            log.warning(f"directory exists: {self._task_data_dir} and not empty. Skipping extraction.")
        else:
            if not os.path.isdir(self._task_data_dir):
                os.mkdir(self._task_data_dir)

            with zipfile.ZipFile(self._task_data_zip, 'r') as zip_ref:
                zip_ref.extractall(self._task_data_dir)
                log.info(f"extracting: {self._task_data_zip}  in {self._task_data_dir} and not empty. Skipping extraction.")
        
    def _read_data(self):
        log.info(f"reading data for task: {self.task_name}")
        self._download_data()
        self._unzip_data()
        # process the data into a dict according to the task        
        getattr(self, "_read_{0}".format(self.task_name))()
        
    def _get_desc(self):
        desc = ""
        if os.path.isfile(self._task_data_desc):
            with open(self._task_data_desc, 'r') as f:
                desc = f.read()
        return desc

    def _read_haspeede(self):
        #log.info ("reading HaSpeeDe 2 data")
        training_file = os.path.join(
            self._task_data_dir,
            "anonimizzati", 
            "Development_set", 
            "haspeede2_dev_taskAB_anon_revised.tsv"
        )
        with open(training_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.training_set.append({
                    "id": row["id"],
                    "text": row["full_text"],
                    "label": eval(row["hs"])
                })
        test_file = os.path.join(
            self._task_data_dir,
            "anonimizzati", 
            "Test_set", 
            "haspeede2_reference_taskAB-tweets_anon_revised.tsv"
        )
        with open(test_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.test_set.append({
                    "id": row["id"],
                    "text": row["full_text"],
                    "label": eval(row["hs"])
                })
        # set target metadata
        self.target_values = [0,1]
        self.target_desc = ["not hateful", "hateful"]
        self.feature_info = ["text: tweets or news headlines"]

    def _read_eventi(self):
        # read training data FBK part (already tokenized)
        training_path = os.path.join(
            self._task_data_dir,
            "EVENTI_ELG",
            "EVENTI_CORPUS",
            "Training-EVENTI-FBK/",
            "training-data",
            "*"
        )
        for xml_file in glob(os.path.join(training_path, "*.xml")):
            document = ElementTree.parse(xml_file)
            root = document.getroot()
            # read events
            events = dict()
            for event in root.find("Markables").findall("EVENT"):
                for i, token_anchor in enumerate(event.findall("token_anchor")):
                    if i == 0:
                        prefix = "B-"
                    else:
                        prefix = "I-"
                    if event.attrib["class"]!="":
                        events[token_anchor.attrib["t_id"]] = prefix+event.attrib["class"]
                    else:
                        events[token_anchor.attrib["t_id"]] = "O"

            # read tokens
            tokens = []
            labels = []
            sentence_id = None
            for token in root.findall("token"):
                if token.attrib["sentence"] != sentence_id:
                    if len(tokens) > 0:
                        self.training_set.append({
                            "id": root.attrib["doc_name"]+"_"+sentence_id,
                            "tokens": tokens,
                            "labels": labels
                        })
                        tokens = []
                        labels = []
                                          
                tokens.append(token.text)
                if token.attrib["t_id"] in events:
                    labels.append(events[token.attrib["t_id"]])
                else:
                    labels.append("O")
                sentence_id = token.attrib["sentence"] 
            self.training_set.append({
                "id": root.attrib["doc_name"]+"_"+sentence_id,
                "tokens": tokens,
                "labels": labels
            })

        # read training data ILC part (not tokenized)
        training_path = os.path.join(
            self._task_data_dir,
            "EVENTI_ELG",
            "EVENTI_CORPUS",
            "Training-EVENTI-ILC/"
        )
        for xml_file in glob(os.path.join(training_path, "*.xml")):
            # workaround because some files are not valid XML
            with open(xml_file) as f:
                xml = f.read().replace("&", "&amp;")
            root = ElementTree.fromstring(xml)
            
            # read events
            events = dict()
            for event in root.find("TAGS").findall("EVENT"):
                events[eval(event.attrib["start"])] = "B-"+event.attrib["class"]
            
            # read text
            tokens = []
            sentences = root.find("TEXT").text.strip().split("\n")
            for sentence in sentences:
                tokens.append(sentence.split(" "))
            # compute offsets to match the event tags
            offsets = []
            offset = 0
            for sentence in tokens:
                offset_sentence = []
                for token in sentence:
                    offset_sentence.append(offset)
                    offset += (len(token)+1)
                offsets.append(offset_sentence)
    
            for s, sentence in enumerate(tokens):
                labels = []
                for offset in offsets[s]:
                    if offset in events:
                        labels.append(events[offset])        
                    else:
                        labels.append("O")
                self.training_set.append({
                    "id": os.path.basename(xml_file),
                    "tokens": sentence,
                    "labels": labels
                })

        # read test data 
        test_path = os.path.join(
            self._task_data_dir,
            "EVENTI_ELG",
            "EVENTI_Task_EVALITA_2014_Test-Gold_MAIN",
            "Gold-Main",
            "Gold-Main_taskABCD"
        )
        for xml_file in glob(os.path.join(test_path, "*.xml")):
            document = ElementTree.parse(xml_file)
            root = document.getroot()
            # read events
            events = dict()
            for event in root.find("Markables").findall("EVENT"):
                for i, token_anchor in enumerate(event.findall("token_anchor")):
                    if i == 0:
                        prefix = "B-"
                    else:
                        prefix = "I-"
                    if event.attrib["class"]!="":
                        events[token_anchor.attrib["t_id"]] = prefix+event.attrib["class"]
                    else:
                        events[token_anchor.attrib["t_id"]] = "O"

            # read tokens
            tokens = []
            labels = []
            sentence_id = None
            for token in root.findall("token"):
                if token.attrib["sentence"] != sentence_id:
                    if len(tokens) > 0:
                        self.test_set.append({
                            "id": root.attrib["doc_name"]+"_"+sentence_id,
                            "tokens": tokens,
                            "labels": labels
                        })
                        tokens = []
                        labels = []
                                          
                if not token.text is None:
                    tokens.append(token.text)
                else:
                    tokens.append("")
                if token.attrib["t_id"] in events:
                    labels.append(events[token.attrib["t_id"]])
                else:
                    labels.append("O")
                sentence_id = token.attrib["sentence"] 
            self.test_set.append({
                "id": root.attrib["doc_name"]+"_"+sentence_id,
                "tokens": tokens,
                "labels": labels
            })
        # set target metadata
        self.target_values = ["B-ASPECTUAL", "B-I_ACTION", "B-I_STATE", "B-PERCEPTION", "B-REPORTING", "B-STATE", "B-OCCURRENCE", "I-ASPECTUAL", "I-I_ACTION", "I-I_STATE", "I-PERCEPTION", "I-REPORTING", "I-STATE", "I-OCCURRENCE", "O"]
        self.target_desc = ["token at the beginning of event that codes information on a particular aspect in the description of another event", "token at the beginning of intensional action", "token at the beginning of event that denotes stative situations", "token at the beginning of event involving physical perception of another event", "token at the beginning of action of an entity declaring something", "token at the beginning of circumstance in which something obtains or holds true", "token at the beginning of other type of event describing situations that occur in the world", "token inside event that codes information on a particular aspect in the description of another event", "tokeninside intensional action", "token inside event that denotes stative situations", "token inside event involving physical perception of another event", "token inside action of an entity declaring something", "token inside circumstance in which something obtains or holds true", "token inside other type of event describing situations that occur in the world", "no event"]
        self.feature_info = ["list of tokens: news articles and stories annotated with temporal information at different levels"]

    def _read_facta(self):
        # read training data FBK part (already tokenized)
        training_path = os.path.join(
            self._task_data_dir,
            "facta_ELG",
            "main_training_FactA-Fact-Ita-Bank",
            "*"
        )
        for xml_file in glob(os.path.join(training_path, "*.xml")):
            document = ElementTree.parse(xml_file)
            root = document.getroot()
            # read events
            events = dict()
            for event in root.find("Markables").findall("EVENT"):
                for i, token_anchor in enumerate(event.findall("token_anchor")):
                    if i == 0:
                        prefix = "B-"
                    else:
                        prefix = "I-"
                    if event.attrib["certainty"] == "":
                        events[token_anchor.attrib["t_id"]] = prefix+"UNDERSPECIFIED"
                    else:
                        events[token_anchor.attrib["t_id"]] = prefix+event.attrib["certainty"]
            # read tokens
            tokens = []
            labels = []
            sentence_id = None
            for token in root.findall("token"):
                if token.attrib["sentence"] != sentence_id:
                    if len(tokens) > 0:
                        self.training_set.append({
                            "id": root.attrib["doc_name"]+"_"+sentence_id,
                            "tokens": tokens,
                            "labels": labels
                        })
                        tokens = []
                        labels = []
                                          
                tokens.append(token.text)
                if token.attrib["t_id"] in events:
                    labels.append(events[token.attrib["t_id"]])
                else:
                    labels.append("O")
                sentence_id = token.attrib["sentence"] 
            self.training_set.append({
                "id": root.attrib["doc_name"]+"_"+sentence_id,
                "tokens": tokens,
                "labels": labels
            })

        
        # read test data 
        test_path = os.path.join(
            self._task_data_dir,
            "facta_ELG",
            "main_testing_witac_facta_evalita2016",
            "corpus_*"
        )
        for xml_file in glob(os.path.join(test_path, "*.xml")):
            document = ElementTree.parse(xml_file)
            root = document.getroot()
            # read events
            events = dict()
            for event in root.find("Markables").findall("EVENT"):
                for i, token_anchor in enumerate(event.findall("token_anchor")):
                    if i == 0:
                        prefix = "B-"
                    else:
                        prefix = "I-"
                    if event.attrib["certainty"] == "":
                        events[token_anchor.attrib["t_id"]] = prefix+"UNDERSPECIFIED"
                    else:
                        events[token_anchor.attrib["t_id"]] = prefix+event.attrib["certainty"]

            # read tokens
            tokens = []
            labels = []
            sentence_id = None
            for token in root.findall("token"):
                if token.attrib["sentence"] != sentence_id:
                    if len(tokens) > 0:
                        self.test_set.append({
                            "id": root.attrib["doc_name"]+"_"+sentence_id,
                            "tokens": tokens,
                            "labels": labels
                        })
                        tokens = []
                        labels = []
                                          
                tokens.append(token.text)
                if token.attrib["t_id"] in events:
                    labels.append(events[token.attrib["t_id"]])
                else:
                    labels.append("O")
                sentence_id = token.attrib["sentence"] 
            self.test_set.append({
                "id": root.attrib["doc_name"]+"_"+sentence_id,
                "tokens": tokens,
                "labels": labels
            })
        # set target metadata
        self.target_values = ["B-CERTAIN", "B-NON_CERTAIN", "B-UNDERSPECIFIED", "I-CERTAIN", "I-NON_CERTAIN", "I-UNDERSPECIFIED", "O"]
        self.target_desc = ["certain about the event - beginning of sequence", "not certain about the event - beginning of sequence", "certainty not specified - beginning of sequence", "certain about the event - inside of sequence", "not certain about the event - inside of sequence", "certainty not specified - inside of sequence", "no event"]
        self.feature_info = ["list of tokens: news stories, Wikinews articles or tweets annotated with event factuality information"]

    def _read_sentipolc(self):
        training_file = os.path.join(
            self._task_data_dir,
            "SENTIPOLC16_ELG", 
            "training_set_sentipolc16_anon_rev.csv"
        )
        labelmap = {
            "00": "neutral",
            "01": "negative",
            "10": "positive",
            "11": "mixed"
        }
        with open(training_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            columns = reader.fieldnames
            for row in reader:
                self.training_set.append({
                    "id": row["idtwitter"],
                    "text": row["text"],
                    "label": labelmap[row["opos"]+row["oneg"]]
                })
        test_file = os.path.join(
            self._task_data_dir,
            "SENTIPOLC16_ELG", 
            "test_set_sentipolc16_gold2000_anon_rev.csv"
        )
        with open(test_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";", fieldnames=columns)
            for row in reader:
                self.test_set.append({
                    "id": row["idtwitter"],
                    "text": row["text"],
                    "label": labelmap[row["opos"]+row["oneg"]]
                })
        # set target metadata
        self.target_values = ["neutral", "positive", "negative", "mixed"]
        self.target_desc = ["no sentiment", "positive", "negative", "both positive and negative"]
        self.feature_info = ["text: tweets"]
             
    def _read_textualentailment(self):
        training_file = os.path.join(
            self._task_data_dir,
            "Official_v1.1",
            "dev.xml"
        )
        document = ElementTree.parse(training_file)
        corpus = document.getroot()
        for pair in corpus.findall("pair"):
            self.training_set.append({
                "id": pair.attrib["id"],
                "label": int(pair.attrib["entailment"]=="YES"),
                "text1": pair.find("h").text,
                "text2": pair.find("t").text
            })            
        test_file = os.path.join(
            self._task_data_dir,
            "Official_v1.1",
            "test_gold.xml"
        )
        document = ElementTree.parse(test_file)
        corpus = document.getroot()
        for pair in corpus.findall("pair"):
            self.test_set.append({
                "id": pair.attrib["id"],
                "label": int(pair.attrib["entailment"]=="YES"),
                "text1": pair.find("h").text,
                "text2": pair.find("t").text
            })
        # set target metadata
        self.target_values = [0,1]
        self.target_desc = ["no entailment", "entailment"]
        self.feature_info = ["hypothesis", "text"]

    def _read_ironita(self):
        training_file = os.path.join(
            self._task_data_dir,
            "Ironita_ELG", 
            "training_ironita2018_anon_REV_.csv"
        )
        with open(training_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                self.training_set.append({
                    "id": row["id"],
                    "text": row["text"],
                    "label": eval(row["irony"])
                })
        test_file = os.path.join(
            self._task_data_dir,
            "Ironita_ELG", 
            "test_gold_ironita2018_anon_REV_.csv"
        )
        with open(test_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                self.test_set.append({
                    "id": row["id"],
                    "text": row["text"],
                    "label": eval(row["irony"])
                })
        # set target metadata
        self.target_values = [0, 1]
        self.target_desc = ["not ironic", "ironic"]
        self.feature_info = ["text: tweets"]

class Task:
    def __init__(self, _task_name):
        if not _task_name in tasks:
            log.error(f"task {_task_name} not in task list")
            raise KeyError (f"task {_task_name} not in task list")
        self.task_name = _task_name
        self.task_type = tasks[self.task_name]["task"]
        self.data = DataReader(self.task_name)
        self.desc = self.data._get_desc()
        self.link = "https://live.european-language-grid.eu/catalogue/corpus/"+str(tasks[self.task_name]["id"])
        
    def evaluate(self, predictions):
        # TODO fix return evalulation metrics
        return getattr(self, "_eval_{0}".format(self.task_type))(predictions)
        
    def _eval_classification(self, predictions):
        # TODO: add special case for SENTIPOLC
        report = classification_report(
            [item["label"] for item in self.data.test_set], 
            predictions,
            output_dict=True)
        if self.task_name == "sentipolc":
            sentipolc_pos_label = {
                "neutral": 0,
                "negative": 0,
                "positive": 1,
                "mixed": 1
            }
            report_positive = classification_report(
                [sentipolc_pos_label[item["label"]] for item in self.data.test_set], 
                [sentipolc_pos_label[pred] for pred in predictions],
                output_dict=True)
            sentipolc_neg_label = {
                "neutral": 0,
                "negative": 1,
                "positive": 0,
                "mixed": 1
            }
            report_negative = classification_report(
                [sentipolc_neg_label[item["label"]] for item in self.data.test_set], 
                [sentipolc_neg_label[pred] for pred in predictions],
                output_dict=True)

            result = {
                "precision_neutral": report["neutral"]["precision"],
                "recall_neutral": report["neutral"]["recall"],
                "f1_neutral": report["neutral"]["f1-score"],
                "precision_negative": report["negative"]["precision"],
                "recall_negative": report["negative"]["recall"],
                "f1_negative": report["negative"]["f1-score"],
                "precision_positive": report["positive"]["precision"],
                "recall_positive": report["positive"]["recall"],
                "f1_positive": report["positive"]["f1-score"],
                "precision_mixed": report["mixed"]["precision"],
                "recall_mixed": report["mixed"]["recall"],
                "f1_mixed": report["mixed"]["f1-score"],
                "precision_sentipolc_pos": report_positive["macro avg"]["precision"],
                "recall_sentipolc_pos": report_positive["macro avg"]["recall"],
                "f1_sentipolc_pos": report_positive["macro avg"]["f1-score"],
                "precision_sentipolc_neg": report_negative["macro avg"]["precision"],
                "recall_sentipolc_neg": report_negative["macro avg"]["recall"],
                "f1_sentipolc_neg": report_negative["macro avg"]["f1-score"],
                "precision_sentipolc": (report_positive["macro avg"]["precision"]+report_negative["macro avg"]["precision"])/2,
                "recall_sentipolc": (report_positive["macro avg"]["recall"]+report_negative["macro avg"]["recall"])/2,
                "f1_sentipolc": (report_positive["macro avg"]["f1-score"]+report_negative["macro avg"]["f1-score"])/2,
                "accuracy": report["accuracy"]
            }
        else:
            result = {
                "precision_0": report["0"]["precision"],
                "recall_0": report["0"]["recall"],
                "f1_0": report["0"]["f1-score"],
                "precision_1": report["1"]["precision"],
                "recall_1": report["1"]["recall"],
                "f1_1": report["1"]["f1-score"],
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
                "accuracy": report["accuracy"]
            }
        return result
    
    def _eval_pairs(self, predictions):
        report = classification_report(
            [item["label"] for item in self.data.test_set], 
            predictions,
            output_dict=True)
        return report
    
    def _eval_sequence(self, predictions):
        # accuracy
        hits = 0
        total = 0
        for i, sentence in enumerate(predictions):
            gold_labels = self.data.test_set[i]["labels"]
            pred_labels = [list(token.items())[0][1] for token in sentence]
            for g, p in zip(gold_labels, pred_labels):
                total += 1
                if g==p:
                    hits += 1
        return {'accuracy': hits/total}
                
