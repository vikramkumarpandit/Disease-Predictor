function generateSymptomInputs() {
  const num = parseInt(document.getElementById("numSymptoms").value);
  const container = document.getElementById("symptomInputs");
  const form = document.getElementById("symptomForm");
  const error = document.getElementById("error");
  const result = document.getElementById("result");

  container.innerHTML = "";
  result.hidden = true;
  error.hidden = true;

  if (isNaN(num) || num < 1) {
    error.innerText = "⚠️ Please enter a valid number of symptoms.";
    error.hidden = false;
    return;
  }

  for (let i = 1; i <= num; i++) {
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = `Symptom ${i}`;
    input.className = "symptom-input";
    container.appendChild(input);
  }

  form.hidden = false;
}

async function submitSymptoms() {
  const inputs = document.getElementsByClassName("symptom-input");
  const symptoms = [];
  const error = document.getElementById("error");
  const result = document.getElementById("result");

  for (let input of inputs) {
    let v = input.value.trim().toLowerCase();
    if (v) symptoms.push(v);
  }

  if (symptoms.length === 0) {
    error.innerText = "⚠️ Please fill in at least one symptom.";
    error.hidden = false;
    result.hidden = true;
    return;
  }

  console.log("➡️ Sending symptoms:", symptoms);

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms })
    });
    const data = await resp.json();
    console.log("⬅️ Received:", data);

    if (data.error) {
      error.innerText = `❌ ${data.error}`;
      error.hidden = false;
      result.hidden = true;
      return;
    }

    // Build table rows
    const tbody = document.querySelector("#resultTable tbody");
    tbody.innerHTML = "";
    (data.predictions || []).forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.Disease}</td>
        <td>${Number(row.Chances).toFixed(6)}</td>
        <td>${row.Specialist}</td>
        <td>${row.Description}</td>
      `;
      tbody.appendChild(tr);
    });

    result.hidden = false;
    error.hidden = true;
  } catch (e) {
    console.error(e);
    error.innerText = "⚠️ Server not responding.";
    error.hidden = false;
    result.hidden = true;
  }
}
