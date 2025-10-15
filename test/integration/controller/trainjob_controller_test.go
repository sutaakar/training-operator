/*
Copyright 2024 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"fmt"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"
	jobsetconsts "sigs.k8s.io/jobset/pkg/constants"
	schedulerpluginsv1alpha1 "sigs.k8s.io/scheduler-plugins/apis/scheduling/v1alpha1"

	trainer "github.com/kubeflow/trainer/pkg/apis/trainer/v1alpha1"
	"github.com/kubeflow/trainer/pkg/constants"
	jobsetplgconsts "github.com/kubeflow/trainer/pkg/runtime/framework/plugins/jobset/constants"
	testingutil "github.com/kubeflow/trainer/pkg/util/testing"
	"github.com/kubeflow/trainer/test/integration/framework"
	"github.com/kubeflow/trainer/test/util"
)

var _ = ginkgo.Describe("TrainJob controller", ginkgo.Ordered, func() {
	var ns *corev1.Namespace

	resRequests := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("1"),
		corev1.ResourceMemory: resource.MustParse("4Gi"),
	}

	ginkgo.BeforeAll(func() {
		fwk = &framework.Framework{}
		cfg = fwk.Init()
		ctx, k8sClient = fwk.RunManager(cfg, true)
	})
	ginkgo.AfterAll(func() {
		fwk.Teardown()
	})

	ginkgo.BeforeEach(func() {
		ns = &corev1.Namespace{
			TypeMeta: metav1.TypeMeta{
				APIVersion: corev1.SchemeGroupVersion.String(),
				Kind:       "Namespace",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "trainjob-",
			},
		}
		gomega.Expect(k8sClient.Create(ctx, ns)).To(gomega.Succeed())
	})

	ginkgo.When("Reconciling TrainJob", func() {
		var (
			trainJob        *trainer.TrainJob
			trainJobKey     client.ObjectKey
			trainingRuntime *trainer.TrainingRuntime
		)

		ginkgo.AfterEach(func() {
			gomega.Expect(k8sClient.DeleteAllOf(ctx, &trainer.TrainJob{}, client.InNamespace(ns.Name))).Should(gomega.Succeed())
		})

		ginkgo.BeforeEach(func() {
			trainJob = testingutil.MakeTrainJobWrapper(ns.Name, "alpha").
				Suspend(true).
				RuntimeRef(trainer.GroupVersion.WithKind(trainer.TrainingRuntimeKind), "alpha").
				SpecLabel("testingKey", "testingVal").
				SpecAnnotation("testingKey", "testingVal").
				Trainer(
					testingutil.MakeTrainJobTrainerWrapper().
						Container("test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
						Obj()).
				Initializer(
					testingutil.MakeTrainJobInitializerWrapper().
						DatasetInitializer(
							testingutil.MakeTrainJobDatasetInitializerWrapper().
								StorageUri("hf://trainjob-dataset").
								Obj(),
						).
						ModelInitializer(
							testingutil.MakeTrainJobModelInitializerWrapper().
								StorageUri("hf://trainjob-model").
								Obj(),
						).
						Obj(),
				).
				Obj()
			trainJobKey = client.ObjectKeyFromObject(trainJob)

			trainingRuntime = testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").
				RuntimeSpec(
					testingutil.MakeTrainingRuntimeSpecWrapper(testingutil.MakeTrainingRuntimeWrapper(metav1.NamespaceDefault, "alpha").Spec).
						WithMLPolicy(
							testingutil.MakeMLPolicyWrapper().
								WithNumNodes(100).
								Obj(),
						).
						PodGroupPolicyCoscheduling(&trainer.CoschedulingPodGroupPolicySource{ScheduleTimeoutSeconds: ptr.To[int32](100)}).
						Container(constants.ModelInitializer, constants.ModelInitializer, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
						Container(constants.DatasetInitializer, constants.DatasetInitializer, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
						Container(constants.Node, constants.Node, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
						Obj()).
				Obj()
		})

		ginkgo.Context("Integration tests for the PlainML Runtime", func() {
			ginkgo.It("Should succeed to create TrainJob with TrainingRuntime", func() {
				ginkgo.By("Creating TrainingRuntime and TrainJob")
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if the appropriate JobSet and PodGroup are created")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(jobSet).Should(gomega.BeComparableTo(
						testingutil.MakeJobSetWrapper(ns.Name, trainJobKey.Name).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Suspend(true).
							Label("testingKey", "testingVal").
							Annotation("testingKey", "testingVal").
							PodLabel(schedulerpluginsv1alpha1.PodGroupLabel, trainJobKey.Name).
							Replicas(1, constants.Node, constants.DatasetInitializer, constants.ModelInitializer).
							Parallelism(1, constants.DatasetInitializer, constants.ModelInitializer).
							Completions(1, constants.DatasetInitializer, constants.ModelInitializer).
							NumNodes(100).
							Container(constants.DatasetInitializer, constants.DatasetInitializer, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
							Env(constants.DatasetInitializer, constants.DatasetInitializer,
								[]corev1.EnvVar{
									{
										Name:  jobsetplgconsts.InitializerEnvStorageUri,
										Value: "hf://trainjob-dataset",
									},
								}...,
							).
							Container(constants.ModelInitializer, constants.ModelInitializer, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
							Env(constants.ModelInitializer, constants.ModelInitializer,
								[]corev1.EnvVar{
									{
										Name:  jobsetplgconsts.InitializerEnvStorageUri,
										Value: "hf://trainjob-model",
									},
								}...,
							).
							Container(constants.Node, constants.Node, "test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Obj(),
						util.IgnoreObjectMetadata))
					pg := &schedulerpluginsv1alpha1.PodGroup{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, pg)).Should(gomega.Succeed())
					g.Expect(pg).Should(gomega.BeComparableTo(
						testingutil.MakeSchedulerPluginsPodGroup(ns.Name, trainJobKey.Name).
							MinMember(102). // 102 replicas = 100 Trainer nodes + 2 Initializers.
							MinResources(corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("102"), // 100 CPUs for Trainer + 2 CPUs for Initializer.
								corev1.ResourceMemory: resource.MustParse("408Gi"),
							}).
							SchedulingTimeout(100).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Obj(),
						util.IgnoreObjectMetadata))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})

			ginkgo.It("Should succeeded to update JobSet only when TrainJob is suspended", func() {
				ginkgo.By("Creating TrainingRuntime and suspended TrainJob")
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet and PodGroup are created")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, &jobsetv1alpha2.JobSet{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, trainJobKey, &schedulerpluginsv1alpha1.PodGroup{})).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Updating suspended TrainJob Trainer image")
				updatedImageName := "updated-trainer-image"
				originImageName := *trainJob.Spec.Trainer.Image
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, trainJob)).Should(gomega.Succeed())
					trainJob.Spec.Trainer.Image = &updatedImageName
					g.Expect(k8sClient.Update(ctx, trainJob)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Trainer image should be updated")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(jobSet).Should(gomega.BeComparableTo(
						testingutil.MakeJobSetWrapper(ns.Name, trainJobKey.Name).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Suspend(true).
							Label("testingKey", "testingVal").
							Annotation("testingKey", "testingVal").
							PodLabel(schedulerpluginsv1alpha1.PodGroupLabel, trainJobKey.Name).
							Replicas(1, constants.Node, constants.DatasetInitializer, constants.ModelInitializer).
							Parallelism(1, constants.DatasetInitializer, constants.ModelInitializer).
							Completions(1, constants.DatasetInitializer, constants.ModelInitializer).
							NumNodes(100).
							Container(constants.DatasetInitializer, constants.DatasetInitializer, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
							Env(constants.DatasetInitializer, constants.DatasetInitializer,
								[]corev1.EnvVar{
									{
										Name:  jobsetplgconsts.InitializerEnvStorageUri,
										Value: "hf://trainjob-dataset",
									},
								}...,
							).
							Container(constants.ModelInitializer, constants.ModelInitializer, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
							Env(constants.ModelInitializer, constants.ModelInitializer,
								[]corev1.EnvVar{
									{
										Name:  jobsetplgconsts.InitializerEnvStorageUri,
										Value: "hf://trainjob-model",
									},
								}...,
							).
							Container(constants.Node, constants.Node, updatedImageName, []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Obj(),
						util.IgnoreObjectMetadata))
					pg := &schedulerpluginsv1alpha1.PodGroup{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, pg)).Should(gomega.Succeed())
					g.Expect(pg).Should(gomega.BeComparableTo(
						testingutil.MakeSchedulerPluginsPodGroup(ns.Name, trainJobKey.Name).
							MinMember(102).
							MinResources(corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("102"), // 100 CPUs for Trainer + 2 CPUs for Initializer.
								corev1.ResourceMemory: resource.MustParse("408Gi"),
							}).
							SchedulingTimeout(100).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Obj(),
						util.IgnoreObjectMetadata))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Unsuspending TrainJob")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, trainJob)).Should(gomega.Succeed())
					trainJob.Spec.Suspend = ptr.To(false)
					g.Expect(k8sClient.Update(ctx, trainJob)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(ptr.Deref(jobSet.Spec.Suspend, false)).Should(gomega.BeFalse())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Trying to restore Trainer image")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, trainJob)).Should(gomega.Succeed())
					trainJob.Spec.Trainer.Image = &originImageName
					g.Expect(k8sClient.Update(ctx, trainJob)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet keep having updated Trainer image")
				gomega.Consistently(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					for _, rJob := range jobSet.Spec.ReplicatedJobs {
						if rJob.Name == constants.Node {
							g.Expect(rJob.Template.Spec.Template.Spec.Containers[0].Image).Should(gomega.Equal(updatedImageName))
						}
					}
				}, util.ConsistentDuration, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Trying to re-suspend TrainJob and restore Trainer image")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, trainJob))
					trainJob.Spec.Suspend = ptr.To(true)
					trainJob.Spec.Trainer.Image = &originImageName
					g.Expect(k8sClient.Update(ctx, trainJob)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet image is restored")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(jobSet.Spec.Suspend).ShouldNot(gomega.BeNil())
					g.Expect(*jobSet.Spec.Suspend).Should(gomega.BeTrue())
					for _, rJob := range jobSet.Spec.ReplicatedJobs {
						if rJob.Name == constants.Node {
							g.Expect(rJob.Template.Spec.Template.Spec.Containers[0].Image).Should(gomega.Equal(originImageName))
						}
					}
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})
		})

		ginkgo.Context("Integration tests for the Torch Runtime", func() {
			ginkgo.It("Should succeed to create TrainJob with Torch TrainingRuntime", func() {
				ginkgo.By("Creating Torch TrainingRuntime and TrainJob")
				trainJob = testingutil.MakeTrainJobWrapper(ns.Name, "alpha").
					RuntimeRef(trainer.GroupVersion.WithKind(trainer.TrainingRuntimeKind), "alpha").
					Trainer(
						testingutil.MakeTrainJobTrainerWrapper().
							Container("test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Env([]corev1.EnvVar{{Name: "TRAIN_JOB", Value: "value"}}...).
							Obj()).
					Obj()
				trainJobKey = client.ObjectKeyFromObject(trainJob)

				trainingRuntime = testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").
					RuntimeSpec(
						testingutil.MakeTrainingRuntimeSpecWrapper(testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").Spec).
							WithMLPolicy(
								testingutil.MakeMLPolicyWrapper().
									WithNumNodes(100).
									WithMLPolicySource(*testingutil.MakeMLPolicySourceWrapper().
										TorchPolicy(ptr.To(intstr.FromString("auto")), nil).
										Obj(),
									).
									Obj(),
							).
							Container(constants.Node, constants.Node, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
							Obj()).
					Obj()
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if the appropriate JobSet is created")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(jobSet).Should(gomega.BeComparableTo(
						testingutil.MakeJobSetWrapper(ns.Name, trainJobKey.Name).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Suspend(false).
							Replicas(1, constants.Node, constants.DatasetInitializer, constants.ModelInitializer).
							Parallelism(1, constants.DatasetInitializer, constants.ModelInitializer).
							Completions(1, constants.DatasetInitializer, constants.ModelInitializer).
							NumNodes(100).
							Container(constants.Node, constants.Node, "test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							ContainerTrainerPorts([]corev1.ContainerPort{{ContainerPort: constants.ContainerTrainerPort, Protocol: "TCP"}}).
							Env(constants.Node, constants.Node,
								[]corev1.EnvVar{
									{
										Name:  "TRAIN_JOB",
										Value: "value",
									},
									{
										Name:  constants.TorchEnvNumNodes,
										Value: "100",
									},
									{
										Name:  constants.TorchEnvNumProcPerNode,
										Value: "1",
									},
									{
										Name: constants.TorchEnvNodeRank,
										ValueFrom: &corev1.EnvVarSource{
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: constants.JobCompletionIndexFieldPath,
											},
										},
									},
									{
										Name:  constants.TorchEnvMasterAddr,
										Value: fmt.Sprintf("alpha-%s-0-0.alpha", constants.Node),
									},
									{
										Name:  constants.TorchEnvMasterPort,
										Value: fmt.Sprintf("%d", constants.ContainerTrainerPort),
									},
								}...,
							).
							Obj(),
						util.IgnoreObjectMetadata))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})

			ginkgo.It("Should succeeded to reconcile TrainJob conditions with Complete condition", func() {
				ginkgo.By("Creating TrainingRuntime and suspended TrainJob")
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet and PodGroup are created")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, &jobsetv1alpha2.JobSet{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, trainJobKey, &schedulerpluginsv1alpha1.PodGroup{})).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if TrainJob has Suspended and Created conditions")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionTrue,
							Reason:  trainer.TrainJobSuspendedReason,
							Message: constants.TrainJobSuspendedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the TrainJob has Resumed and Created conditions after unsuspended")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					gotTrainJob.Spec.Suspend = ptr.To(false)
					g.Expect(k8sClient.Update(ctx, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Updating the JobSet condition with Completed")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					meta.SetStatusCondition(&jobSet.Status.Conditions, metav1.Condition{
						Type:    string(jobsetv1alpha2.JobSetCompleted),
						Reason:  jobsetconsts.AllJobsCompletedReason,
						Message: jobsetconsts.AllJobsCompletedMessage,
						Status:  metav1.ConditionTrue,
					})
					g.Expect(k8sClient.Status().Update(ctx, jobSet)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the TranJob has Resumed, Created, and Completed conditions")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
						{
							Type:    trainer.TrainJobComplete,
							Status:  metav1.ConditionTrue,
							Reason:  jobsetconsts.AllJobsCompletedReason,
							Message: jobsetconsts.AllJobsCompletedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})

			ginkgo.It("Should succeeded to reconcile TrainJob conditions with Failed condition", func() {
				ginkgo.By("Creating TrainingRuntime and suspended TrainJob")
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet and PodGroup are created")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, &jobsetv1alpha2.JobSet{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, trainJobKey, &schedulerpluginsv1alpha1.PodGroup{})).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Unsuspending the TrainJob")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					gotTrainJob.Spec.Suspend = ptr.To(false)
					g.Expect(k8sClient.Update(ctx, gotTrainJob)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Waiting for TrainJob Created=True and Suspended=False condition")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Updating the JobSet condition with Failed")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					meta.SetStatusCondition(&jobSet.Status.Conditions, metav1.Condition{
						Type:    string(jobsetv1alpha2.JobSetFailed),
						Reason:  jobsetconsts.FailedJobsReason,
						Message: jobsetconsts.FailedJobsMessage,
						Status:  metav1.ConditionTrue,
					})
					g.Expect(k8sClient.Status().Update(ctx, jobSet)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the TranJob has Resumed, Created, and Failed conditions")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
						{
							Type:    trainer.TrainJobFailed,
							Status:  metav1.ConditionTrue,
							Reason:  jobsetconsts.FailedJobsReason,
							Message: jobsetconsts.FailedJobsMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})

			ginkgo.It("Should succeed to create TrainJob with PodSpecOverrides", func() {
				ginkgo.By("Creating Torch TrainingRuntime and TrainJob")
				trainJob = testingutil.MakeTrainJobWrapper(ns.Name, "alpha").
					RuntimeRef(trainer.GroupVersion.WithKind(trainer.TrainingRuntimeKind), "alpha").
					Trainer(
						testingutil.MakeTrainJobTrainerWrapper().
							Container("test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Env([]corev1.EnvVar{{Name: "TRAIN_JOB", Value: "value"}}...).
							Obj()).
					PodSpecOverrides([]trainer.PodSpecOverride{
						{
							TargetJobs:         []trainer.PodSpecOverrideTargetJob{{Name: constants.Node}},
							ServiceAccountName: ptr.To("override-sa"),
							InitContainers: []trainer.ContainerOverride{
								{
									Name: "override-init-container",
									Env: []corev1.EnvVar{
										{
											Name:  "INIT_ENV",
											Value: "override_init",
										},
										{
											Name:  "NEW_VALUE",
											Value: "from_overrides",
										},
									},
								},
							},
						},
					}).
					Obj()
				trainJobKey = client.ObjectKeyFromObject(trainJob)

				trainingRuntime = testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").
					RuntimeSpec(
						testingutil.MakeTrainingRuntimeSpecWrapper(testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").Spec).
							WithMLPolicy(
								testingutil.MakeMLPolicyWrapper().
									WithNumNodes(100).
									WithMLPolicySource(*testingutil.MakeMLPolicySourceWrapper().
										TorchPolicy(ptr.To(intstr.FromString("auto")), nil).
										Obj(),
									).
									Obj(),
							).
							InitContainer(constants.Node, "override-init-container", "test:runtime", []corev1.EnvVar{
								{
									Name:  "INIT_ENV",
									Value: "original_init",
								},
								{
									Name:  "DATASET_PATH",
									Value: "runtime",
								},
							}...,
							).
							Container(constants.Node, constants.Node, "test:runtime", []string{"runtime"}, []string{"runtime"}, resRequests).
							Obj()).
					Obj()
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if the appropriate JobSet is created")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(jobSet).Should(gomega.BeComparableTo(
						testingutil.MakeJobSetWrapper(ns.Name, trainJobKey.Name).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Suspend(false).
							Replicas(1, constants.Node, constants.DatasetInitializer, constants.ModelInitializer).
							Parallelism(1, constants.DatasetInitializer, constants.ModelInitializer).
							Completions(1, constants.DatasetInitializer, constants.ModelInitializer).
							NumNodes(100).
							ServiceAccountName(constants.Node, "override-sa").
							InitContainer(constants.Node, "override-init-container", "test:runtime",
								corev1.EnvVar{
									Name:  "INIT_ENV",
									Value: "override_init",
								},
								corev1.EnvVar{
									Name:  "NEW_VALUE",
									Value: "from_overrides",
								},
								corev1.EnvVar{
									Name:  "DATASET_PATH",
									Value: "runtime",
								},
							).
							Container(constants.Node, constants.Node, "test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							ContainerTrainerPorts([]corev1.ContainerPort{{ContainerPort: constants.ContainerTrainerPort, Protocol: "TCP"}}).
							Env(constants.Node, constants.Node,
								[]corev1.EnvVar{
									{
										Name:  "TRAIN_JOB",
										Value: "value",
									},
									{
										Name:  constants.TorchEnvNumNodes,
										Value: "100",
									},
									{
										Name:  constants.TorchEnvNumProcPerNode,
										Value: "1",
									},
									{
										Name: constants.TorchEnvNodeRank,
										ValueFrom: &corev1.EnvVarSource{
											FieldRef: &corev1.ObjectFieldSelector{
												FieldPath: constants.JobCompletionIndexFieldPath,
											},
										},
									},
									{
										Name:  constants.TorchEnvMasterAddr,
										Value: fmt.Sprintf("alpha-%s-0-0.alpha", constants.Node),
									},
									{
										Name:  constants.TorchEnvMasterPort,
										Value: fmt.Sprintf("%d", constants.ContainerTrainerPort),
									},
								}...,
							).
							Obj(),
						util.IgnoreObjectMetadata))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})
		})

		ginkgo.Context("Integration Tests for the OpenMPI Runtime", func() {
			var (
				cmKey  client.ObjectKey
				secKey client.ObjectKey
			)
			ginkgo.It("Should succeed to create TrainJob with OpenMPI TrainingRuntime", func() {
				ginkgo.By("Creating OpenMPI TrainingRuntime and TrainJob")
				trainJob = testingutil.MakeTrainJobWrapper(ns.Name, "alpha").
					RuntimeRef(trainer.GroupVersion.WithKind(trainer.TrainingRuntimeKind), "alpha").
					Trainer(
						testingutil.MakeTrainJobTrainerWrapper().
							NumNodes(2).
							Container("test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Env([]corev1.EnvVar{{Name: "TRAIN_JOB", Value: "value"}}...).
							Obj()).
					Obj()
				trainJobKey = client.ObjectKeyFromObject(trainJob)
				cmKey = client.ObjectKey{
					Name:      fmt.Sprintf("%s%s", trainJobKey.Name, constants.MPIHostfileConfigMapSuffix),
					Namespace: trainJobKey.Namespace,
				}
				secKey = client.ObjectKey{
					Name:      fmt.Sprintf("%s%s", trainJobKey.Name, constants.MPISSHAuthSecretSuffix),
					Namespace: trainJobKey.Namespace,
				}

				trainingRuntime = testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").
					RuntimeSpec(
						testingutil.MakeTrainingRuntimeSpecWrapper(testingutil.MakeTrainingRuntimeWrapper(ns.Name, "alpha").Spec).
							LauncherReplica().
							Replicas(1, constants.Launcher).
							WithMLPolicy(
								testingutil.MakeMLPolicyWrapper().
									WithNumNodes(1).
									WithMLPolicySource(*testingutil.MakeMLPolicySourceWrapper().
										MPIPolicy(ptr.To[int32](8), ptr.To(trainer.MPIImplementationOpenMPI), ptr.To("/root/.ssh"), ptr.To(false)).
										Obj(),
									).
									Obj(),
							).
							Container(constants.Node, constants.Node, "test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Obj()).
					Obj()
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if the appropriate JobSet is created")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					g.Expect(jobSet).Should(gomega.BeComparableTo(
						testingutil.MakeJobSetWrapper(ns.Name, trainJobKey.Name).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Suspend(false).
							LauncherReplica().
							Replicas(1, constants.Node, constants.DatasetInitializer, constants.ModelInitializer, constants.Launcher).
							Parallelism(1, constants.DatasetInitializer, constants.ModelInitializer, constants.Launcher).
							Completions(1, constants.DatasetInitializer, constants.ModelInitializer, constants.Launcher).
							NumNodes(2).
							Container(constants.Node, constants.Node, "test:trainjob", []string{"trainjob"}, []string{"trainjob"}, resRequests).
							Volumes(constants.Launcher,
								corev1.Volume{
									Name: constants.MPISSHAuthVolumeName,
									VolumeSource: corev1.VolumeSource{
										Secret: &corev1.SecretVolumeSource{
											SecretName: fmt.Sprintf("%s%s", trainJobKey.Name, constants.MPISSHAuthSecretSuffix),
											Items: []corev1.KeyToPath{
												{
													Key:  corev1.SSHAuthPrivateKey,
													Path: constants.MPISSHPrivateKeyFile,
												},
												{
													Key:  constants.MPISSHPublicKey,
													Path: constants.MPISSHPublicKeyFile,
												},
												{
													Key:  constants.MPISSHPublicKey,
													Path: constants.MPISSHAuthorizedKeys,
												},
											},
										},
									},
								},
								corev1.Volume{
									Name: constants.MPIHostfileVolumeName,
									VolumeSource: corev1.VolumeSource{
										ConfigMap: &corev1.ConfigMapVolumeSource{
											LocalObjectReference: corev1.LocalObjectReference{
												Name: fmt.Sprintf("%s%s", trainJobKey.Name, constants.MPIHostfileConfigMapSuffix),
											},
											Items: []corev1.KeyToPath{{
												Key:  constants.MPIHostfileName,
												Path: constants.MPIHostfileName,
												Mode: ptr.To[int32](0444),
											}},
										},
									},
								},
							).
							Volumes(constants.Node,
								corev1.Volume{
									Name: constants.MPISSHAuthVolumeName,
									VolumeSource: corev1.VolumeSource{
										Secret: &corev1.SecretVolumeSource{
											SecretName: fmt.Sprintf("%s%s", trainJobKey.Name, constants.MPISSHAuthSecretSuffix),
											Items: []corev1.KeyToPath{
												{
													Key:  corev1.SSHAuthPrivateKey,
													Path: constants.MPISSHPrivateKeyFile,
												},
												{
													Key:  constants.MPISSHPublicKey,
													Path: constants.MPISSHPublicKeyFile,
												},
												{
													Key:  constants.MPISSHPublicKey,
													Path: constants.MPISSHAuthorizedKeys,
												},
											},
										},
									},
								},
							).
							VolumeMounts(constants.Launcher, constants.Node,
								corev1.VolumeMount{Name: constants.MPISSHAuthVolumeName, MountPath: "/root/.ssh"},
								corev1.VolumeMount{Name: constants.MPIHostfileVolumeName, MountPath: constants.MPIHostfileDir},
							).
							VolumeMounts(constants.Node, constants.Node,
								corev1.VolumeMount{Name: constants.MPISSHAuthVolumeName, MountPath: "/root/.ssh"},
							).
							Env(constants.Launcher, constants.Node,
								corev1.EnvVar{
									Name:  constants.OpenMPIEnvHostFileLocation,
									Value: fmt.Sprintf("%s/%s", constants.MPIHostfileDir, constants.MPIHostfileName),
								},
								corev1.EnvVar{
									Name:  constants.OpenMPIEnvKeepFQDNHostNames,
									Value: "true",
								},
								corev1.EnvVar{
									Name:  constants.OpenMPIEnvDefaultSlots,
									Value: "8",
								},
								corev1.EnvVar{
									Name:  constants.OpenMPIEnvKeyRSHArgs,
									Value: constants.OpenMPIEnvDefaultValueRSHArgs,
								},
							).
							Env(constants.Node, constants.Node,
								corev1.EnvVar{
									Name:  "TRAIN_JOB",
									Value: "value",
								},
							).
							Obj(),
						util.IgnoreObjectMetadata))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the appropriate ConfigMap is created")
				gomega.Eventually(func(g gomega.Gomega) {
					cm := &corev1.ConfigMap{}
					g.Expect(k8sClient.Get(ctx, cmKey, cm)).To(gomega.Succeed())
					g.Expect(cm).Should(gomega.BeComparableTo(
						testingutil.MakeConfigMapWrapper(cmKey.Name, cmKey.Namespace).
							WithData(map[string]string{
								constants.MPIHostfileName: `alpha-node-0-0.alpha slots=8
alpha-node-0-1.alpha slots=8
`,
							}).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Obj(),
						util.IgnoreObjectMetadata, cmp.Comparer(testingutil.MPISecretDataComparer)))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the appropriate Secret is created")
				gomega.Eventually(func(g gomega.Gomega) {
					sec := &corev1.Secret{}
					g.Expect(k8sClient.Get(ctx, secKey, sec)).To(gomega.Succeed())
					g.Expect(sec).Should(gomega.BeComparableTo(
						testingutil.MakeSecretWrapper(secKey.Name, secKey.Namespace).
							WithImmutable(true).
							WithData(map[string][]byte{
								corev1.SSHAuthPrivateKey:  []byte("EXIST"),
								constants.MPISSHPublicKey: []byte("EXIST"),
							}).
							WithType(corev1.SecretTypeSSHAuth).
							ControllerReference(trainer.SchemeGroupVersion.WithKind(trainer.TrainJobKind), trainJobKey.Name, string(trainJob.UID)).
							Obj(),
						util.IgnoreObjectMetadata, cmp.Comparer(testingutil.MPISecretDataComparer)))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})

			ginkgo.It("Should succeeded to reconcile TrainJob conditions with Complete condition", func() {
				ginkgo.By("Creating TrainingRuntime and suspended TrainJob")
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet, ConfigMap, and Secret are created")
				gomega.Expect(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, &jobsetv1alpha2.JobSet{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, cmKey, &corev1.ConfigMap{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, secKey, &corev1.Secret{})).Should(gomega.Succeed())
				})

				ginkgo.By("Checking if TrainJob has Suspended and Created conditions")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionTrue,
							Reason:  trainer.TrainJobSuspendedReason,
							Message: constants.TrainJobSuspendedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the TrainJob has Resumed and Created conditions after unsuspended")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					gotTrainJob.Spec.Suspend = ptr.To(false)
					g.Expect(k8sClient.Update(ctx, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Updating the JobSet condition with Completed")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					meta.SetStatusCondition(&jobSet.Status.Conditions, metav1.Condition{
						Type:    string(jobsetv1alpha2.JobSetCompleted),
						Reason:  jobsetconsts.AllJobsCompletedReason,
						Message: jobsetconsts.AllJobsCompletedMessage,
						Status:  metav1.ConditionTrue,
					})
					g.Expect(k8sClient.Status().Update(ctx, jobSet)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the TranJob has Resumed, Created, and Completed conditions")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
						{
							Type:    trainer.TrainJobComplete,
							Status:  metav1.ConditionTrue,
							Reason:  jobsetconsts.AllJobsCompletedReason,
							Message: jobsetconsts.AllJobsCompletedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})

			ginkgo.It("Should succeeded to reconcile TrainJob conditions with Failed condition", func() {
				ginkgo.By("Creating TrainingRuntime and suspended TrainJob")
				gomega.Expect(k8sClient.Create(ctx, trainingRuntime)).Should(gomega.Succeed())
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(trainingRuntime), trainingRuntime)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
				gomega.Expect(k8sClient.Create(ctx, trainJob)).Should(gomega.Succeed())

				ginkgo.By("Checking if JobSet, ConfigMap, and Secret are created")
				gomega.Eventually(func(g gomega.Gomega) {
					g.Expect(k8sClient.Get(ctx, trainJobKey, &jobsetv1alpha2.JobSet{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, cmKey, &corev1.ConfigMap{})).Should(gomega.Succeed())
					g.Expect(k8sClient.Get(ctx, secKey, &corev1.Secret{})).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Unsuspending the TrainJob")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					gotTrainJob.Spec.Suspend = ptr.To(false)
					g.Expect(k8sClient.Update(ctx, gotTrainJob)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Waiting for TrainJob Created=True and Suspended=False condition")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Updating the JobSet condition with Failed")
				gomega.Eventually(func(g gomega.Gomega) {
					jobSet := &jobsetv1alpha2.JobSet{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, jobSet)).Should(gomega.Succeed())
					meta.SetStatusCondition(&jobSet.Status.Conditions, metav1.Condition{
						Type:    string(jobsetv1alpha2.JobSetFailed),
						Reason:  jobsetconsts.FailedJobsReason,
						Message: jobsetconsts.FailedJobsMessage,
						Status:  metav1.ConditionTrue,
					})
					g.Expect(k8sClient.Status().Update(ctx, jobSet)).Should(gomega.Succeed())
				}, util.Timeout, util.Interval).Should(gomega.Succeed())

				ginkgo.By("Checking if the TranJob has Resumed, Created, and Failed conditions")
				gomega.Eventually(func(g gomega.Gomega) {
					gotTrainJob := &trainer.TrainJob{}
					g.Expect(k8sClient.Get(ctx, trainJobKey, gotTrainJob)).Should(gomega.Succeed())
					g.Expect(gotTrainJob.Status.Conditions).Should(gomega.BeComparableTo([]metav1.Condition{
						{
							Type:    trainer.TrainJobSuspended,
							Status:  metav1.ConditionFalse,
							Reason:  trainer.TrainJobResumedReason,
							Message: constants.TrainJobResumedMessage,
						},
						{
							Type:    trainer.TrainJobFailed,
							Status:  metav1.ConditionTrue,
							Reason:  jobsetconsts.FailedJobsReason,
							Message: jobsetconsts.FailedJobsMessage,
						},
					}, util.IgnoreConditions))
				}, util.Timeout, util.Interval).Should(gomega.Succeed())
			})
		})
	})
})
