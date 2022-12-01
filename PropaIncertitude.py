import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import lhsmdu

nb_echan = 100 #Nombre d'échantillons
np.random.seed(0)

#Taille des fibres
moy = 6
ecart = 0.6
serie = np.random.normal(moy,ecart,nb_echan)

#Porosité

plt.figure("PDF")
#Tracé des histogrammes normalisés
plot_pdf = plt.subplot(1,2,1)
plot_cdf = plt.subplot(1,2,2)

plt.subplots_adjust(hspace=0.5)

plot_pdf.hist(serie,25,density=True)


plot_pdf.set_xlabel("Epaisseur MCP")
plot_pdf.set_ylabel("Nombre")


#Tracer et fitter les distributions normales correspondantes
xmin_df,xmax_df = moy-4*ecart, moy+4*ecart
lnspc_df = np.linspace(xmin_df,xmax_df,len(serie))
fit_moy_df,fit_ecart_df = stats.norm.fit(serie)

#Superposition des PDF

pdf_df = stats.norm.pdf(lnspc_df,fit_moy_df,fit_ecart_df)
label = "Moyenne ="+"{:.2f}".format(fit_moy_df)+'\n'+"Ecart-type ="+"{:.2f}".format(fit_ecart_df)
plot_pdf.plot(lnspc_df,pdf_df,label=label)

#Tracé des CDF
plot_cdf.hist(serie,20,cumulative=True,density=True)


plot_cdf.set_xlabel("e MCP")
plot_cdf.set_ylabel("Probabilité")


cdf = stats.norm.cdf(lnspc_df,fit_moy_df,fit_ecart_df)

plot_cdf.plot(lnspc_df,cdf,label="Norm")

#Légende et plot

plot_pdf.set_title("PDF porosité")
plot_cdf.set_title("CDF df")

plot_pdf.legend()
plot_cdf.legend()

plt.show()

#MonteCarlo

MC = lhsmdu.createRandomStandardUniformMatrix(1,100)
MC = stats.norm.ppf(MC,fit_moy_df,fit_ecart_df)

print(MC)