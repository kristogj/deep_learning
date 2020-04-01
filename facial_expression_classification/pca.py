from dataloader import load_data, balanced_sampler
import numpy as np
import matplotlib.pyplot as plt
import random

from constants import TASK


class PCA:

    def __init__(self, num_components):
        """

        Args:
            num_components: Number of principal components we would like to use
        """
        self.num_components = num_components
        self.mean = None
        self.principal_components = None
        self.img_dims = None
        self.sqrt_eigenvalues = None

    def fit(self, data):
        """
        STEP 1: Subtract the mean face from every face
        STEP 2: Construct the matrix we will find eigenvectors of (L = A^TA)
        Step 3: Find the M eigenvectors of L
        Step 4: Only keep the given number of principal components
        Step 5: Sanity check

        Args:
            data: numpy array of images N x height x width

        Returns:
            None
        """
        N, height, width = data.shape
        self.img_dims = (height, width)

        # Change each image to a big vector
        data = data.reshape(N, -1)

        # STEP 1: Subtract the mean face from every face
        self.mean = np.mean(data, axis=0)
        data = data - self.mean

        # STEP 2: Construct the matrix we will find eigenvectors of (L = A^TA
        L = np.dot(data, data.T) / N  # Shape should be N x N where N is number of images

        # Step 3: Find the M eigenvectors of L
        eigen_values, eigen_vectors = np.linalg.eigh(L)
        # Sort eigenvectors based on their eigenvalues
        sorted_indexes = np.argsort(eigen_values)[::-1]  # returns an array of indices that would sort the array
        eigen_vectors, eigen_values = eigen_vectors[:, sorted_indexes], eigen_values[sorted_indexes]

        # Step 4: Only keep the given number of principal components
        eigen_vectors = eigen_vectors[:, :self.num_components]
        eigen_values = eigen_values[:self.num_components]

        # Turk and Pentlands Trick
        temp = np.dot(data.T, eigen_vectors)
        temp = temp / np.linalg.norm(temp, axis=0)

        # STEP 5: Sanity check
        project = np.dot(temp[:, 0], data.T)
        assert np.isclose(np.mean(project, axis=0), 0)
        assert np.isclose(np.std(project, axis=0), np.sqrt(eigen_values[0]))

        # Save this for later, will be used to normalize the input vector as described in PCA Power Point
        self.sqrt_eigenvalues = np.sqrt(eigen_values.reshape(1, -1))

        # This is what we will use then training the logistic and softmax regression model
        self.principal_components = temp

    def transform(self, data):
        """
        Transform images into vectors with self.num_components dimensions
        Args:
            data: Single image

        Returns:

        """
        # Reshape data to a long vector
        data = data.reshape(data.shape[0], -1)

        # Subtract mean of training set images
        data = data - self.mean

        # Project the data onto the principal components
        data = np.dot(data, self.principal_components) / self.sqrt_eigenvalues

        return data

    def transform_inverse(self, data):
        """
        Takes the reduced images and inverse them back to the original dimension
        Args:
            data: One pca transform image of shape 1 x num_components

        Returns:
            images of original self.img_dims
        """
        images = np.dot(data * self.sqrt_eigenvalues, self.principal_components.T)  # pc shape 43008 x num_components
        images += self.mean
        return images.reshape((data.shape[0], self.img_dims[0], self.img_dims[1]))

    def display_pc(self, settings):
        images = self.principal_components.T.reshape(self.num_components, self.img_dims[0], self.img_dims[1])
        images = images[:4, :, :]
        images = np.concatenate(images, axis=1)
        plt.imshow(images, cmap='gray')
        plt.title('Showing the first {} principal components'.format(4))
        plt.savefig("./graphs/display_pca_{}.png".format(settings[TASK]))
        plt.show()


def show(image, title="", save_path=None):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    if save_path:
        plt.savefig("./graphs/{}".format(save_path))
    plt.show()


def main():
    # Example run of how to use PCA
    path = "./aligned"
    emotions = ['sadness', 'happiness', 'anger', 'disgust', 'fear', 'surprise']
    data, cnt = load_data(path)
    balanced_subset = balanced_sampler(data, cnt, emotions)

    # Images would be your training data
    images = []
    for e in emotions:
        images.extend(balanced_subset[e])

    random.shuffle(images)
    images = np.array(images[:])  # This would be the images in the training set

    pca = PCA(10)
    pca.fit(images)

    images = images[:10]

    # Transform some of the images
    index = 0
    show(images[index])
    projected_image = pca.transform(images)
    inverse = pca.transform_inverse(projected_image)
    show(inverse[index])


# For testing purposes only
if __name__ == '__main__':
    main()
