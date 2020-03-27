from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file=open("model.pkl","rb")
clf=pickle.load(file)
file.close()
@app.route('/', methods=["GET","POST"])
def hello_world():
    if (request.method=="POST"):
        akdict=request.form
        fever=int(akdict["fever"])
        age = int(akdict["age"])
        pain=int(akdict["pain"])
        runnynose = int(akdict["runnynose"])
        diffbreath = int(akdict["diffbreath"])
        inputFeatures = [fever,pain,age,runnynose,diffbreath]
        infprob = clf.predict_proba([inputFeatures])[0][1]
        #print(infprob)
        return render_template("show.html", inf=round(infprob*100))
    return render_template("index.html")
    #return "The infection percentage is "+str(infprob*100)
    
if __name__=="__main":
    app.run(debug=True)
