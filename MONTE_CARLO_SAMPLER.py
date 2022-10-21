import numpy as np
import matplotlib.pyplot as plt

class GenerateDataset():

    def __init__(self, t1, t2, t3, t4, A1, A2, A3, A4, bounds_low, bounds_high, numSamples):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.low = bounds_low
        self.high = bounds_high
        self.numSamples = numSamples

        self.residuals = self.sampler()

    def pdf(self, t):
        tr = 0.8 # rise time in ns 
        return (self.A1 * (np.exp( -( t/self.t1 )) - np.exp(-( t/tr ) ) )  / ( self.t1 - tr ) ) \
            +  (self.A2 * (np.exp( -( t/self.t2 ) ) - np.exp( -( t/tr ) )) / ( self.t2 - tr ) ) \
            +  (self.A3 * (np.exp( -( t/self.t3 ) ) - np.exp( -( t/tr ) )) / ( self.t3 - tr ) ) \
            +  (self.A4 * (np.exp( -( t/self.t4 ) ) - np.exp( -( t/tr ) )) / ( self.t4 - tr ) )

    def sampler(self):
        
        # compute the max of the PDF for given t1 and t2
        t_scan = np.linspace(self.low, self.high, 10000)
        pdf_scan = self.pdf(t_scan)
        MAX_PDF = max(pdf_scan)

        residuals = []
        for iSample in range(self.numSamples):
            # print(iSample)
            # sample t position uniformly within bounds 
            t_sample = np.random.uniform(low = self.low, high = self.high)

            # sample uniformly value of "encapsulation function": uniform between 0---> max(PDF)
            u_sample = np.random.uniform(low = 0, high = MAX_PDF)
            # check if the new sample is under or above the PDF
            pdf_val = self.pdf(t_sample)

            if u_sample <= pdf_val:
                residuals.append(t_sample)
            else:
                while u_sample > pdf_val:
                    t_sample = np.random.uniform(low = self.low, high = self.high)
                    pdf_val = self.pdf(t_sample)
                    u_sample = np.random.uniform(low = 0, high = MAX_PDF)
                residuals.append(t_sample)
            
        return residuals