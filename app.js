require("dotenv").config();
const express = require('express');
const mongoose = require("mongoose")

// 
const path = require("path"); 

const router = express.Router();
const app  = express();
const port = process.env.PORT || 5000;

// express middleware
// app.use(express.urlencoded({extended: true}));
app.use(express.json());
const cors = require('cors');
app.use(cors());
// mongodb
require("./db/connection");
const User = require("./Model/user");
const UserFeedback = require("./Model/userfeedback");

// Model

// Routes
// app.get("/", (req,res)=>{
//     res.send("index")
// });


// Route to handle login requests
app.post("/login", (req, res) => {
    // Log the request body for debugging
    console.log("Request body:", req.body);
    
    const { email, password } = req.body;
    
    // Find the user by email
    User.findOne({ email: email })
        .then(user => {
            if (user) {
                if (password === user.password) {
                    // Send success message if password matches
                    // res.status(200).json({ message: "Login Successful", user: user });
                    res.send({ message: "Login Successful", user:user})
                    console.log("Login Successful");
                    
                } else {
                    // Send error message if password doesn't match
                    // res.status(401).json({ message: "Password didn't match" });
                    res.send({ message: "Password didn't match" })
                    console.log("Password didn't match");
                   
                }
            } else {
                // Send error message if user not found
                // res.status(404).json({ message: "User not found" });
                res.send({ message: "User not found" })
                console.log("User not found");
               
            }
        })
        .catch(err => {
            // Handle any unexpected errors
            res.send({ message: "Error during login" })
            console.error("Error during login:");
        
            // res.status(500).json({ message: "Internal Server Error" });
        });
});


// Routes Signup
app.post("/signup", (req,res)=>{
    // res.send("My Api Signup")
    console.log(req.body);
    const {name, email, password} = req.body;
    // check already email registerd
  // Using promises with findOne()
User.findOne({ email: email })
    .then(user => {
        if (user) {
            res.send({ message: "User already registered" });     
            console.log("User Already Register");
          alert("User Already Register");
        } else {
            // Create a new user
            const newUser = new User({
                name: name,
                email: email,
                password: password
            });

            // Save the new user
            return newUser.save();
        }
    })
    .then(savedUser => {
        // Handle saved user
        // For example, you might want to send a success response
        res.send({ message: "User registered successfully. Please Login Now" });
        console.log("User registered successfully");
       
    })
    .catch(error => {
        // Handle any errors that occur during the process
        // For example, database errors or validation errors
        // res.status(500).send({ error: error.message });
        console.log("Error");
    });

});
// Routes Feedback
app.post("/feedback", (req,res)=>{
    // res.send("My Api Signup")
    console.log(req.body);
    const {query,response, referencedDocument, feedback1,relaventAnswer,feedback2} = req.body;
    // check already email registerd
  // Using promises with findOne()
  UserFeedback.findOne({ feedback1: feedback1 })
    .then(feedback => {
        if (feedback) {
           
            res.send({ message: "feedback Sent Succussfully" });     
            
          alert("feedback Sent Succussfully");
        } else {
            // Create a new user
            const newUser1 = new UserFeedback({
                query: query,
                referencedDocument: referencedDocument,
                feedback1: feedback1,
                relaventAnswer:relaventAnswer,
                feedback2:feedback2

            });

            // Save the new user
            return newUser1.save();
        }
    })
    .then(savedUser => {
        // Handle saved user
        // For example, you might want to send a success response
        res.send({ message: "feedback Sent Succussfully" });
        console.log("feedback Sent Succussfully");
       
    })
    .catch(error => {
        // Handle any errors that occur during the process
        // For example, database errors or validation errors
        // res.status(500).send({ error: error.message });
        console.log("Error");
    });

});

//
// app.get("/", (req, res) => { 
//     app.use(express.static(path.resolve(__dirname, "frontend", "build"))); 
//     res.sendFile(path.resolve(__dirname, "frontend", "build", "index.html")); 
//   }); 
//

app.listen(port, ()=>{
    console.log(`Example app listening on port ${port}`);
})
module.exports = router;