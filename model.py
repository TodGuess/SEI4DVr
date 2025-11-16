import numpy as np

class SI4RDVr:
    def __init__(self, *, regions:int, movM:np.ndarray=None, virusFactors:dict, initialSettings:dict):
        self.infRs = virusFactors['infection']
        self.proRs = virusFactors['promotion']
        self.dthRs = virusFactors['death']
        self.rcvLR = virusFactors['recoveryL']
        self.rI, self.vI = virusFactors['recoveryI'], virusFactors['vaccinI']
        self.vcnGR, self.vcnLR = virusFactors['vaccinG'], virusFactors['vaccinL']

        assert 0 <= initialSettings['startH'] <= 23, 'Starting time step must be between 0-23'
        self.startH = initialSettings['startH']
        self.regions, self.movM = regions, movM

        self.I = [[initialSettings['infected'][r],0,0,0] for r in range(regions)]
        self.S = [initialSettings['population'][r] - self.I[r][0] for r in range(regions)]
        self.R = [0]*regions
        self.V = [0]*regions
        self.D = [0]*regions

        self.initial_total = sum(initialSettings['population'])
        self.simT = 0
        self.logList = [np.column_stack([
            self.S,
            [i[0] for i in self.I],
            [i[1] for i in self.I],
            [i[2] for i in self.I],
            [i[3] for i in self.I],
            self.R,
            self.V,
            self.D
        ])]

    def apply_movement(self):
        if self.movM is None:
            return

        n = self.regions
        comps = np.zeros((n, 7))
        comps[:, 0] = self.S
        comps[:, 1:5] = np.array(self.I)
        comps[:, 5] = self.R
        comps[:, 6] = self.V
        pop_total = comps.sum(axis=1)

        M = self.movM[:, :, self.simT % self.movM.shape[2]]
        outflow = M.sum(axis=1)
        scale = np.ones_like(outflow)
        mask_over = outflow > pop_total
        scale[mask_over] = pop_total[mask_over] / outflow[mask_over]
        M_scaled = (M.T * scale).T
        ratio = comps / pop_total[:, None]
        ratio[pop_total==0, :] = 0.0
        comp_out = ratio * M_scaled.sum(axis=1)[:, None]
        comp_in = M_scaled.T.dot(ratio)
        comps_after = comps + comp_in - comp_out
        comps_after = np.maximum(comps_after, 0.0)
        self.S = comps_after[:, 0].tolist()
        self.I = comps_after[:, 1:5].tolist()
        self.R = comps_after[:, 5].tolist()
        self.V = comps_after[:, 6].tolist()

    def updateH(self):
        for reg in range(self.regions):
            Sr, Rr, Vr, Dr = self.S[reg], self.R[reg], self.V[reg], self.D[reg]
            Ir = self.I[reg][:]

            pop = Sr + sum(Ir) + Rr + Vr
            if pop <= 0:
                continue
            newInfR = sum(self.infRs[i] * Ir[i] / pop for i in range(4))
            newDth = [self.dthRs[i] * Ir[i] for i in range(4)]
            infPromo = [self.proRs[i] * Ir[i] for i in range(4)]
            rcvLoss = self.rcvLR * Rr
            newVcn, vcnLoss = self.vcnGR * Sr, self.vcnLR * Vr
            SrDelta = -Sr*newInfR + rcvLoss + vcnLoss - newVcn
            IrDelta = [0, 0, 0, 0]
            for i in range(4):
                if i==0:
                    IrDelta[0] = (Sr + (1-self.rI)*Rr + (1-self.vI)*Vr)*newInfR - infPromo[0] - newDth[0]
                else:
                    IrDelta[i] += infPromo[i-1] - infPromo[i] - newDth[i]
            RrDelta = infPromo[3] - rcvLoss - (1-self.rI)*Rr*newInfR
            VrDelta = newVcn - vcnLoss - (1-self.vI)*Vr*newInfR
            DrDelta = sum(newDth)
            self.S[reg], self.R[reg], self.V[reg], self.D[reg], self.I[reg] = Sr+SrDelta, Rr+RrDelta, Vr+VrDelta, Dr+DrDelta, [Ir[i] + IrDelta[i] for i in range(4)]

        self.apply_movement()
        self.logList.append(np.column_stack([
            self.S,
            [i[0] for i in self.I],
            [i[1] for i in self.I],
            [i[2] for i in self.I],
            [i[3] for i in self.I],
            self.R,
            self.V,
            self.D
        ]))
        
        self.simT += 1
