// Copyright 2023 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pytorch

import (
	"testing"

	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
)

func TestSetPodEnv(t *testing.T) {
	g := gomega.NewWithT(t)

	// Prepare a base PyTorchJob.
	pytorchjob := &kubeflowv1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-job",
			Annotations: map[string]string{
				"checkpoint.config.kubeflow.org/existing-var": "new-value",
				"checkpoint.config.kubeflow.org/new-var":      "new-value",
			},
		},
		Spec: kubeflowv1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec{
				kubeflowv1.PyTorchJobReplicaTypeMaster: {
					Replicas: func(i int32) *int32 { return &i }(1),
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name: "pytorch",
								Ports: []corev1.ContainerPort{{
									Name:          "pytorchjob-port",
									ContainerPort: 23456,
								}},
							}},
						},
					},
				},
				kubeflowv1.PyTorchJobReplicaTypeWorker: {
					Replicas: func(i int32) *int32 { return &i }(1),
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name: "pytorch",
								Ports: []corev1.ContainerPort{{
									Name:          "pytorchjob-port",
									ContainerPort: 23456,
								}},
							}},
						},
					},
				},
			},
		},
	}

	// Case 1: An environment variable from a checkpoint annotation already exists.
	podTemplateSpecWithExistingEnv := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Env: []corev1.EnvVar{
						{Name: "EXISTING_VAR", Value: "original-value"},
					},
				},
			},
		},
	}

	err := setPodEnv(pytorchjob, podTemplateSpecWithExistingEnv, "master", "0")
	g.Expect(err).NotTo(gomega.HaveOccurred())

	// Verify that the existing variable was not overwritten and the new one was added.
	g.Expect(podTemplateSpecWithExistingEnv.Spec.Containers[0].Env).To(gomega.ContainElement(corev1.EnvVar{Name: "EXISTING_VAR", Value: "original-value"}))
	g.Expect(podTemplateSpecWithExistingEnv.Spec.Containers[0].Env).To(gomega.ContainElement(corev1.EnvVar{Name: "NEW_VAR", Value: "new-value"}))

	// Case 2: No conflicting environment variables.
	podTemplateSpecNew := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name: "pytorch",
				Ports: []corev1.ContainerPort{{
					Name:          "pytorchjob-port",
					ContainerPort: 23456,
				}},
			}},
		},
	}
	err = setPodEnv(pytorchjob, podTemplateSpecNew, "master", "0")
	g.Expect(err).NotTo(gomega.HaveOccurred())

	// Verify that the new variable was added.
	g.Expect(podTemplateSpecNew.Spec.Containers[0].Env).To(gomega.ContainElement(corev1.EnvVar{Name: "NEW_VAR", Value: "new-value"}))

	// Case 3: Check for default env vars like PYTHONUNBUFFERED.
	g.Expect(podTemplateSpecNew.Spec.Containers[0].Env).To(gomega.ContainElement(corev1.EnvVar{Name: "PYTHONUNBUFFERED", Value: "1"}))
}
