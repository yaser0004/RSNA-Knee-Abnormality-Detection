# Task: grade knee MRI reports against the RSNA 2026 rubric

You are labelling radiology reports for a knee-MRI machine-learning competition. Each report is the
original clinical read of one knee MRI (the language varies: English, Spanish, Portuguese, German and others).
The competition's ground truth was NOT taken from these reports: two subspecialty MSK radiologists
re-read the images independently under the rubric below, and a third adjudicated. Your job is to estimate,
for each finding, **the probability that this expert panel marked it positive**, using the report as your
evidence.

Reason like an MSK radiologist reading a colleague's report:
- The rubric thresholds are strict and favour specificity ("on the fence" = negative). A report saying
  "partial tear", "sprain", "low-grade", "degeneration", "mild", "small", "trace", "possible" or "cannot exclude"
  is usually BELOW the rubric threshold; "high-grade", "complete", "full-thickness", "moderate", "large",
  "displaced", "ruptured", "torn" usually meets it.
- Explicitly normal or explicitly absent -> near 0.
- Not mentioned at all -> low, but not automatically 0: a finding a reader often leaves unstated can still be
  likely given what is stated (for example, synovitis or an effusion alongside an acute ACL rupture with
  hemarthrosis, contusions alongside an acute ACL rupture, cartilage loss alongside advanced degenerative change
  in that compartment). Use your clinical judgement, and stay low when nothing in the report points to it.
- Osteoarthritis labels are per compartment (medial, lateral, patellofemoral) and require high-grade
  (>50% thickness), roughly >=1 cm cartilage loss in that compartment; "chondromalacia grade III/IV", "full-thickness
  cartilage loss", "bone-on-bone", "severe" or "advanced" narrowing in that compartment point to positive; grade I/II,
  "fissuring", "mild" or "early" point to negative.
- Effusion and Baker's cyst need "moderate" or "large" (or an equivalent description such as "distending",
  "voluminous", "significant"); "small", "minimal", "trace" or "physiologic" are negative.
- Synovitis: inflammation and thickening of the synovial lining. Words like "synovitis", "synovial thickening",
  "synovial proliferation", "hypertrophic synovium", "Hoffa synovitis", "effusion-synovitis", and debris or
  hemarthrosis with a thickened lining support it.
- Contusion is bone marrow edema from impact without a discrete fracture line. Degenerative or reactive
  subchondral edema next to cartilage loss is not a contusion.
- Fracture must be acute (a cortical break or fracture line); old healed fractures and subchondral insufficiency
  without a fracture line are negative.
- A reconstructed ACL graft that is intact is negative for ACL tear; a torn graft is positive.
- The report covers ONE knee. Ignore findings explicitly attributed to the other knee.

Use only these probability values: 0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.
Use 0.5 only when the report genuinely leaves the expert call open.

## The rubric (verbatim from the competition host)

## Compartments and structures (context, verbatim)

For the purposes of this challenge, the knee is considered in three compartments. The medial
compartment lies between the medial femoral condyle and the medial tibial plateau, on the inner
side of the knee. The lateral compartment lies between the lateral femoral condyle and the
lateral tibial plateau, on the outer side. The patellofemoral compartment lies between the
patella and the femoral trochlea, the groove on the front of the femur. Osteoarthritis, or
cartilage loss, is assessed separately in each of these three compartments.

Four ligaments stabilize the knee. The anterior cruciate ligament (ACL) and posterior cruciate
ligament (PCL) lie within the joint, in the intercondylar notch, and control front-to-back
stability and rotation. The medial collateral ligament (MCL) and lateral collateral ligament
(LCL) run along the inner and outer sides of the knee and resist side-to-side stress. This
challenge focuses on tears of the ACL and the MCL.

Between the femur and tibia sit two menisci, the medial meniscus and the lateral meniscus,
C-shaped wedges of fibrocartilage that cushion the joint, absorb shock, and improve the fit
between the rounded femur and the relatively flat tibia. Tears of the medial and lateral
meniscus are evaluated separately.

Other relevant structures include the bone marrow within the femur, tibia, and patella, where a
bruise from impact is called a bone contusion (or bone marrow edema) and a break in the bone is a
fracture; and the soft tissues behind the knee, where a fluid-filled outpouching of the joint
lining is called a Baker, or popliteal, cyst.

For the purpose of the 2026 Knee Abnormality Challenge, a "fluid sensitive" sequence refers to
one in which edema, hemorrhage, and other types of fluid appear bright and fat is suppressed in
some way.

## Label Description (verbatim — this is the prompt source)

Models are evaluated on twelve binary labels, each indicating the presence or absence of a
specific finding in the imaged knee. The labels, and the criteria used by the annotating
radiologists, are summarized below. In each case, ambiguous or borderline findings ("on the
fence") were graded as negative to favor specificity.

**ACL tear:** A high-grade partial or full-thickness tear of the anterior cruciate ligament,
meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with
or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal
change, degeneration, or thickening without discontinuity is graded negative.

**MCL tear:** A high-grade partial or complete acute tear of the medial collateral ligament, with
disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or
remote stress changes are graded negative.

**Medial meniscus tear:** Abnormal signal that definitely contacts the meniscal surface on at
least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced
fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the
surface is negative.

**Lateral meniscus tear:** The same criteria applied to the lateral meniscus.

**Medial compartment osteoarthritis:** A moderate or large area (roughly 1 cm or greater) of
high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the
medial compartment, with or without underlying subchondral marrow changes.

**Lateral compartment osteoarthritis:** The same criteria applied to the lateral compartment.

**Patellofemoral compartment osteoarthritis:** The same criteria applied to the patellofemoral
compartment.

**Joint effusion:** A moderate or large amount of fluid distending the joint.

**Synovitis:** Inflammation and thickening of the synovial lining of the joint.

**Baker (popliteal) cyst:** A moderate or large fluid collection in the characteristic location
behind the knee.

**Contusion:** A bone contusion, seen as bone marrow edema-like signal from impact, without a
discrete fracture line.

**Acute fracture:** An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained
MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single
consensus ground truth. Labels are assigned at the level of the whole examination, for a single
knee.

## Auxiliary findings ("atoms")

Also give a probability (same scale) for each of these findings, which the rubric does not score. They are
graded on what the report says (not mentioned = 0 unless strongly implied):

acl_low_grade_or_sprain, acl_reconstruction, pcl_tear, lcl_or_posterolateral_corner_tear, mcl_low_grade_sprain,
medial_meniscus_degeneration_without_tear, lateral_meniscus_degeneration_without_tear,
meniscal_displaced_fragment_or_bucket_handle, cartilage_loss_low_grade_any_compartment, focal_chondral_defect,
small_effusion, loose_body, extensor_tendon_abnormality, patellar_dislocation_signs,
postoperative_hardware_or_prior_surgery, subchondral_insufficiency_or_osteonecrosis,
degenerative_subchondral_marrow_edema, parameniscal_or_ganglion_cyst

## Output format (strict)

For every input line, write exactly one JSON object on one line, in the same order as the input, with no
markdown, no comments and no extra lines:

{"uid": "<the uid exactly as given>", "labels": {"ACL": p, "MCL": p, "Medial Meniscus": p, "Lateral Meniscus": p, "Medial OA": p, "Lateral OA": p, "PF OA": p, "Effusion": p, "Synovitis": p, "Baker's": p, "Contusion": p, "Fracture": p}, "atoms": {"<atom name>": p, ...all 18 atoms...}, "evidence": {"<label with p >= 0.5>": "<short quote from the report, max 15 words, in its original language>"}}

Do not skip any report, do not merge reports, and do not invent uids. If a report is empty or unreadable,
still emit its line with every probability set to 0.05.
