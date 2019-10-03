from tkinter import *
import joblib

logreg = joblib.load('dibetic.pkl')


fields = ( 'pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age')

def predict(entries):
   # period rate:
   a = (float(entries['pregnant'].get())) 
   b = (float(entries['glucose'].get())) 
   c = (float(entries['bp'].get())) 
   d = (float(entries['insulin'].get())) 
   e = (float(entries['bmi'].get())) 
   f = (float(entries['pedigree'].get())) 
   g = (float(entries['age'].get())) 
   new_test_data = [a,b,c,d,e,f,g]
   new_y_pred=logreg.predict([new_test_data])
   if(new_y_pred == [0]):
      var = StringVar()
      label = Label( root, textvariable = var, relief = GROOVE  )

      var.set("the person is not a dibetic")
      label.pack()  
    
   else:
      var = StringVar()
      label = Label( root, textvariable = var, relief = RAISED )

      var.set("the person is  dibetic")
      label.pack()
    


def makeform(root, fields):
   entries = {}
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=22, text=field+": ", anchor='w')
      ent = Entry(row)
      ent.insert(0,"0")
      row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
      lab.pack(side = LEFT)
      ent.pack(side = RIGHT, expand = YES, fill = X)
      entries[field] = ent
      
   return entries


   


if __name__ == '__main__':
   root = Tk()
   
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e = ents: fetch(e)))
   b1 = Button(root, text = 'predict',
      command=(lambda e = ents: predict(e)))
   b1.pack(side = LEFT, padx = 5, pady = 5)
  
   b2 = Button(root, text = 'Quit', command = root.destroy)
   b2.pack(side = LEFT, padx = 5, pady = 5)

   
   root.mainloop()
