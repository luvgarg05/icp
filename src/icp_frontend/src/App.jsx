import { useState } from 'react';
import { icp_backend } from 'declarations/icp_backend';

function App() {
  const [greeting, setGreeting] = useState('');

  function handleSubmit(event) {
    event.preventDefault();
    const name = event.target.elements.name.value;
    icp_backend.greet(name).then((greeting) => {
      setGreeting(greeting);
    });
    return false;
  }

  return (
    <main>
      <img src="/logo2.svg" alt="DFINITY logo" />
      <br />
      <br />
      
      {/* Text greeting form */}
      <form action="#" onSubmit={handleSubmit}>
        <label htmlFor="name">Enter your name:&nbsp;</label>
        <input id="name" alt="Name" type="text" />
        <button type="submit">Click Me 😊!</button>
      </form>
      <section id="greeting">{greeting}</section>
  
      <br /><br />
  
      {/* Image upload form */}
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          const formData = new FormData(e.target);
          const res = await fetch('http://localhost:8080/upload', {
            method: 'POST',
            body: formData
          });
          const data = await res.json();
          console.log('Server Response:', data);
        }}
        encType="multipart/form-data"
      >
        <label htmlFor="image">Upload an image: </label>
        <input id="image" name="image" type="file" accept="image/*" />
        <button type="submit">Upload Image</button>
      </form>
    </main>
  );
  
}

export default App;
