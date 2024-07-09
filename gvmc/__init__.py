#! venv/bin/python3

# Third party libraries
import numpy as np
import scipy.stats as scist
import scipy.spatial as scisp
import math as mt


class Gvmc_layer(dict):
    def __init__(self, data=None, **kw):
        # Growth vectors
        self["growth_vecs"] = kw.get("growth_vecs", None)
        # Number of growth vectors
        self["n_unique_gvs"] = kw.get("n_unique_gvs", 4)
        # Initial growth value probs
        self["gv_prob_init"] = kw.get("gv_prob_init",
                                      np.ones((self["n_unique_gvs"],1))*(
                                              1.0/self["n_unique_gvs"]))
        # If data do stuff
        if data is not None:
            self["data"] = data
            self.ndata, self.nsamples = data.shape
            self.make_growth_vecs(ngv=self["n_unique_gvs"])
            self.make_kdtree(k=10)
            self.make_local_density()

            # Unscaled growth value probs
            self["gv_prob"] = np.zeros((self["n_unique_gvs"], self.nsamples)) 
            self.make_gv_directional_probabilities()
        else:
            self["data"] = None


    def make_gv_directional_probabilities(self):
        r"""Go through each point and count pts in gv directions"""
        binary_masks = np.ones((self["n_unique_gvs"], self.nsamples))
        for ipt in range(self.nsamples):
            # Get growth vectors
            gvs = self["growth_vecs"][:, :, ipt]
            # Get pts within k nearest radius
            k_nearest_i = self["k_nearest"][:,ipt]
            k = len(k_nearest_i)
            # Make vectors of nearest relative to pt
            pt = self["data"][:, ipt].reshape(self.ndata, 1)
            # K nearest values
            k_nearest = self["data"][:, k_nearest_i]
            # Get relative vector
            k_nearest_relative = k_nearest - pt
            # Loop through each growth vec
            for igv in range(self["n_unique_gvs"]):
                # Get dot product
                dot_prod = np.dot(k_nearest_relative.T, 
                                gvs[igv, :].reshape(self.ndata, 1))
                # Magnitude for scaling
                nearest_mags = np.linalg.norm(k_nearest_relative, axis=0)
                # Scaled dot product
                dot_prod *= nearest_mags.reshape(k,1)**-1.0
                # Get angle between data and growth vector
                degs = np.arccos(dot_prod)*(180/np.pi)
                Ibool = np.zeros_like(nearest_mags, dtype=bool)
                I2bool = np.zeros_like(nearest_mags, dtype=bool) 
                # Where non-negative angles
                I = np.where(dot_prod > 0)[0]
                # Mask mask
                Ibool[I] = True
                # Count if < 45 degree cone in given direction
                I2 = np.where(degs < 45)[0]
                # Mask mask
                I2bool[I2] = True
                # And masks
                Imask = np.logical_and(Ibool, I2bool)
                # Probability based on that
                self["gv_prob"][igv, ipt] = len(np.where(Imask == True)[0])/k
            if np.sum(self["gv_prob"][:, ipt]) > 0.0:
                # Scale to probability
                self["gv_prob"][:, ipt] *= np.sum(self["gv_prob"][:, ipt])**-1.0
 

    def make_kdtree(self, k=10):
        r"""Make KDTree of data for querying"""
        self["KDTree"] = scisp.KDTree(self["data"].T, leafsize=k)


    def make_local_density(self, k=10):
        r"""Get local density of all pts"""
        self["density"] = np.zeros((self.nsamples))
        self["k_nearest"] = np.zeros((k, self.nsamples), dtype=int)
        for i, pt in enumerate(self["data"].T):
            self["density"][i], self["k_nearest"][:,i] = self.genr8_density(
                                                            pt, k=k)


    def genr8_density(self, pt, k=10):
        r"""Calculate local density of point"""
        # Get distances and index of k nearest points
        dists, inds = self["KDTree"].query(pt.T, k=k)
        # Max dist returned last
        radius = dists[-1]
        # Estimate density in ball
        ball_vol = np.pi**(self.ndata/2) / (
                    mt.gamma((self.ndata/2) + 1) * radius**self.ndata)
        # Return density and index of k nearest
        return ((len(inds)/self.nsamples) * ball_vol, np.int32(inds))
        

    def make_growth_vecs(self, ngv=4):
        r"""Make growth vectors"""
        # Use svd basis vecs for growth vecs
        u, _, _ = np.linalg.svd(self["data"])
        # Mirror for opposite direction vecs too
        u_mirror = -1.0*u
        # Number of data points
        nsamples = self["data"].shape[-1]
        # Make growth vec matrix for 1 pt
        gvi_mat = np.vstack([u, u_mirror])
        # Build full growth vec matrix
        gv_mat = gvi_mat.repeat(nsamples).reshape(*gvi_mat.shape, nsamples) 
        self["growth_vecs"] = gv_mat


    def get_gv_probabilities(self, pt, pt_ind=None, k=1):
        r"""For given pt, get growth vector probabilities"""
        # Check if exact match
        if pt_ind is not None:
            return self.make_gv_probabilities(pt_ind)
        # Get distance of pt to known data
        _, k_nearest_inds = self["KDTree"].query(pt.T, k=k)
        if k == 1:
            k_nearest_inds = k_nearest_inds.reshape((1,k))
        # Get probability of k nearest pts
        avg_probs = np.zeros((self["n_unique_gvs"]))
        for ipt in range(k):
            avg_probs += self.make_gv_probabilities(k_nearest_inds[0,ipt])
        return (1.0/k)*avg_probs


    def make_gv_probabilities(self, i, density_weight=4.0):
        r"""Return growth vec probabilites"""
        # Relative density score
        density_score = self["density"][i] / np.max(self["density"])
        # Dampen confidence implied by density score
        density_score *= density_weight
        # Density adjusted probability
        prob = np.min([1.0, density_score])*self["gv_prob_init"][:,0] + \
               np.max([0.0, (1-density_score)])*self["gv_prob"][:,i]
        return prob


    def get_step_size(self, gmvc, pt):
        r"""For given pt, get step size"""
        # Get uq handle
        uq = gmvc.uq
        # Draw one scaling vector from uq
        uq_fracs = uq.rvs(1)
        # Calc step size
        step_size = np.sqrt(np.sum(uq_fracs**2.0))
        # A diff attempt
        step_size = np.abs(uq_fracs).reshape(len(uq_fracs), 1)
        return step_size


class Gvmc_growth_layer(Gvmc_layer):
    def __init__(self, **kw):
        # Do layer init
        Gvmc_layer.__init__(self, **kw)
        # Extra stuff for growth layer
        self["data"] = None


    def update_layer(self, pt):
        r"""For newly generated point update info"""
        if self["data"] is None:
            self["data"] = np.array(pt)
        else:
            self["data"] = np.hstack([self["data"], pt])


class Gvmc(object):
    def __init__(self, data, uq=None, **kw):
        r"""
        :Inputs:
            *data*: {``None``} | :class:`numpy.ndarray`
                NxM data matrix, N is the dimension \& M the unique samples 
        :Versions:
            * 2023-05-08 ``@aburkhea``: v1.0
        """
        # Set input data
        self._data = data
        # Get dimensions of data
        self.ndata, self.nsamples = data.shape
        # Start base layer properties
        self.layers = np.array([Gvmc_layer(data=data, **kw)], dtype=type(Gvmc_layer))
        # Set uq
        self.uq = self.uq_init(uq, **kw)


    def uq_init(self, uq, uq_frac=0.5, **kw):
        r"""Set uq, allow for default"""
        # If no uq provided
        if uq is None:
            # Zero vector
            mean = np.zeros(self.ndata)
            # Identity with fraction along diagonal
            # cov = np.eye(self.ndata)
            cov = np.cov(self._data)*uq_frac
            # Draw from multivariate normal
            return scist.multivariate_normal(mean, cov, allow_singular=True)
        # Set uq if provided
        self.uq = uq

    def genr8_new_layer(self, nsteps):
        r"""Generate new growth layer that inherits from base"""
        # Inherit from base layer
        kw0 = {k:v for k,v in self.layers[0].items() if k in [""]}
        self.layers = np.append(self.layers, Gvmc_growth_layer(**kw0))
        return self.layers[-1]

    def genr8_step(self, pt, i_reflayer=0):
        r"""Take step based off base layer (0th layer)"""
        # Get reference layer
        ref_layer = self.layers[i_reflayer]
        # Get nubmer of unique growth vectors
        ngv = ref_layer["n_unique_gvs"]
        # Get probabilites of growth vecs from this pt
        p_gv = ref_layer.get_gv_probabilities(pt)
        # Choose growth vector
        step_gv = np.random.choice(np.arange(ngv), size=1, p=p_gv/p_gv.sum())
        # Get step size
        step_size = ref_layer.get_step_size(self, pt)
        # Take step
        step = pt + step_size*ref_layer["growth_vecs"][step_gv,:,0].T
        dists, k_inds = ref_layer["KDTree"].query(step.T, k=10)
        dists0, k_inds0 = ref_layer["KDTree"].query(
                                    ref_layer["data"][:,k_inds[0,0]].T, k=10)
        if dists[0,0] > dists0[-1]:
            step = ref_layer["data"][:,k_inds[0,0]].reshape(step.shape)
        # Take step
        return step


    def make_growth_layer(self, nsteps=100):
        r"""Make new growth layer"""
        # Get current number of layers
        nlayers = len(self.layers)
        # Choose initial point
        init_ind = np.random.choice(np.arange(self.nsamples), size=1)
        # Get initial data point
        pt = self._data[:,init_ind]
        # Start new layer
        new_layer = self.genr8_new_layer(nsteps)
        # Take nsteps
        for istep in range(nsteps):
            # Generate new point by taking step
            new_pt = self.genr8_step(pt)
            # Update the new layer with new pt
            new_layer.update_layer(new_pt)
            # Next new pt based on this one
            pt = new_pt
