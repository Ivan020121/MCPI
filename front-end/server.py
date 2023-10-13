from flask import Flask,render_template,request

app = Flask(__name__)

import pymongo

# torch.cuda.set_device(1)
# torch.cuda.set_device(1)

import torch.utils.data

from sppDatapre import *

if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")

# connect database
def connect_db(db_name, coll_name):
    # mongo_conn = pymongo.MongoClient(host='localhost', port=27017, username='liwenhao', password='lwh020121')
    mongo_conn = pymongo.MongoClient(host='47.99.71.176', port=27017, username='root', password='admin')
    # db = mongo_conn.get_database("mcpi_data")  # specify database
    db = mongo_conn.get_database(db_name)  # specify database
    # coll = db.get_collection("compound_node2vec")  # get collection
    coll = db.get_collection(coll_name)  # get collection
    return coll

# select data
def load_data(db_name, compound_coll_name, protein_coll_name, smiles, seq):
    compound_coll = connect_db(db_name, compound_coll_name)
    protein_coll = connect_db(db_name, protein_coll_name)
    compound_result = compound_coll.find_one({"smiles": smiles})
    protein_result = protein_coll.find_one({"sequence": seq})
    c_id = compound_result["_id"]
    p_id = protein_result["_id"]
    dm = pickle.loads(compound_result["distance_matrix"])
    dm = load_tensor(dm, torch.FloatTensor)
    # dm = create_variable(dm)
    fp = pickle.loads(compound_result["finger_print"])
    fp = load_tensor(fp, torch.FloatTensor)
    # fp = create_variable(fp)
    cn2v = pickle.loads(compound_result["node2vec"])
    cn2v = load_tensor(cn2v, torch.FloatTensor)
    # cn2v = create_variable(cn2v)
    pw2v =pickle.loads(protein_result["word2vec"])
    pw2v = load_tensor(pw2v, torch.FloatTensor)
    # pw2v = create_variable(pw2v)
    pn2v = pickle.loads(protein_result["node2vec"])
    pn2v = load_tensor(pn2v, torch.FloatTensor)
    # pn2v = create_variable(pn2v)
    return c_id, p_id, dm, fp, cn2v, pw2v, pn2v

def load_tensor(data, dtype):
    return dtype([data])

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# predict
def predict(db_name, compound_coll_name, protein_coll_name, smiles, seq, model):
    model = torch.load(model, map_location='cpu')
    c_id, p_id, dm, fp, cn2v, pw2v, pn2v = load_data(db_name, compound_coll_name, protein_coll_name, smiles, seq)
    result = model.forward(dm, fp, pw2v, cn2v, pn2v)
    if int(result) >= 0.5:
        return c_id, p_id, 1
    return c_id, p_id, 0

# -d mcpi_database -c compound_data -p protein_data -s MSPLNQSAEGLPQEASNRSLNATETSEAWDPRTLQALKISLAVVLSVITLATVLSNAFVLTTILLTRKLHTPANYLIGSLATTDLLVSILVMPISIAYTITHTWNFGQILCDIWLSSDITCCTASILHLCVIALDRYWAITDALEYSKRRTAGHAATMIAIVWAISICISIPPLFWRQAKAQEEMSDCLVNTSQISYTIYSTCGAFYIPSVLLIILYGRIYRAARNRILNPPSLYGKRFTTAHLITGSAGSSLCSLNSSLHEGHSHSAGSPLFFNHVKIKLADSALERKRISAARERKATKILGIILGAFIICWLPFFVVSLVLPICRDSCWIHPALFDFFTWLGYLNSLINPIIYTVFNEEFRQAFQKIVPFRKAS -i CC[C@@]1(C[C@@H]2C3=CC(=C(C=C3CCN2C[C@H]1CC(C)C)OC)OC)O

@app.route('/index',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/contact',methods=['GET','POST'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        coll = connect_db("mcpi_database", "contact_data")
        coll.insert_one({
            "name": str(name),
            "email": str(email),
            "message": str(message),
        })
        print(name,email,message)
        return render_template('contact.html')

@app.route('/product',methods=['GET','POST'])
def product():
    if request.method == 'GET':
        return render_template('product.html')
    if request.method == 'POST':
        # df = pd.read_csv(file)
        Sequence = request.form.get('Sequence')
        SMILES = request.form.get('SMILES')
        file_p = request.files["file-1[]"]
        file_c = request.files["file-2[]"]
        if file_p and file_c:
            df_p = pd.read_csv(file_p)
            df_p = df_p[df_p.columns[0]]
            df_c = pd.read_csv(file_c)
            df_c = df_c[df_c.columns[0]]
            list = []
            for p in df_p:
                for c in df_c:
                    c_id, p_id, result = predict("mcpi_database", "compound_data", "protein_data", c, p, "model.pkl")
                    if len(p) > 39:
                        p1 = p[:39] + "..."
                    if len(c) > 39:
                        c1 = c[:39] + "..."
                    dic = {"cid": c_id, "pid": p_id, "seq": p1, "sml": c1, "res": result}
                    list.append(dic)
            return render_template('result.html', results=list)
        # prediction = model.forward()
        else:
            list = []
            c_id, p_id, result = predict("mcpi_database", "compound_data", "protein_data", SMILES, Sequence, "model.pkl")
            if len(Sequence) > 39:
                Sequence = Sequence[:39]+"..."
            if len(SMILES) > 39:
                SMILES = SMILES[:39]+"..."
            dic = {"cid": c_id, "pid": p_id, "seq": Sequence, "sml": SMILES, "res": result}
            list.append(dic)
            return render_template('result.html', results=list)

@app.route('/result',methods=['GET','POST'])
def result():
    if request.method == 'GET':
        return render_template('result.html')

@app.route('/about',methods=['GET','POST'])
def about():
    if request.method == 'GET':
        return render_template('about.html')

@app.route('/sign-in',methods=['GET','POST'])
def sign_in():
    if request.method == 'GET':
        return render_template('sign-in.html')

@app.route('/sign-up',methods=['GET','POST'])
def sign_up():
    if request.method == 'GET':
        return render_template('sign-up.html')

if __name__== '__main__':
    app.run()
