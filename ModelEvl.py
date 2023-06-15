from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

class ModelEvaluation:
    
    def modelEvaluation(model, y_test,y_pred , name):
        score=accuracy_score(y_test,y_pred)*100
        conf_m=confusion_matrix(y_test,y_pred)
        report=classification_report(y_test,y_pred)

        print("Score of " + name + " Is : ")
        print(score)
        print("Confusion Matrix of "+name + " Is : ")
        print(conf_m)
        print("Report of "+name +" Is : ")
        print(report)
